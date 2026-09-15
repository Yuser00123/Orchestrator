import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { GoogleGenAI } from '@google/genai';
import { Sandbox } from 'e2b';
import Groq from 'groq-sdk';
import { Mistral } from '@mistralai/mistralai';
import { CohereClient } from 'cohere-ai';
import OpenAI from 'openai';
import admin from 'firebase-admin';

if (!admin.apps.length) {
  admin.initializeApp(); // uses Application Default Credentials on Cloud Run automatically
}
const db = admin.firestore();

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

const PORT = 3000;
const PROJECT_DIR = '/home/user/project';
const SKILLS_DIR = '/home/user/skills';
const UI_SKILLS_INSTRUCTION =
  'Before building or styling a webpage or UI, check /home/user/skills/anthropic/ (via list_files) for frontend-design or web-artifacts-builder guidance, and read the relevant SKILL.md if it applies. ' +
  'If the task involves Supabase (database, auth, storage, edge functions, RLS policies), check /home/user/skills/supabase/ (via list_files) and read the relevant SKILL.md. Since no Supabase MCP server is connected, use its documented fallback: fetch a docs page as markdown by appending .md to its URL via browse_webpage, or use web_search for anything else.';
const FALLBACK_CHAIN = [
  { provider: 'gemini', model: 'gemini-3.6-flash' },
  { provider: 'gemini', model: 'gemini-3.5-flash-lite' },
  { provider: 'gemini', model: 'gemini-3.1-flash-lite' },
  { provider: 'groq', model: 'openai/gpt-oss-120b' },
  { provider: 'groq', model: 'openai/gpt-oss-20b' },
  { provider: 'mistral', model: 'open-mistral-nemo' },
  { provider: 'mistral', model: 'codestral-latest' },
  { provider: 'mistral', model: 'ministral-8b-latest' },
  { provider: 'cohere', model: 'command-r-plus-08-2024' },
  { provider: 'cohere', model: 'command-r-08-2024' },
  { provider: 'openrouter', model: 'openrouter/free' }
];
const E2B_API_KEY = process.env.E2B_API_KEY;
const GITHUB_USERNAME = process.env.GITHUB_USERNAME;
const GITHUB_TOKEN = process.env.GITHUB_TOKEN;
const MAX_TOOL_LOOPS = 20; // safety cap so a stuck agent can't loop forever

function redactSecrets(text) {
  if (text == null) return '';
  let str = typeof text === 'string' ? text : String(text);
  const secrets = [GITHUB_TOKEN, process.env.VERCEL_TOKEN, process.env.CLOUDFLARE_API_TOKEN].filter(Boolean);
  for (const secret of secrets) {
    str = str.replaceAll(secret, '***');
  }
  return str;
}

if (!process.env.GEMINI_API_KEY) console.warn('WARNING: GEMINI_API_KEY is not set');
if (!E2B_API_KEY) console.warn('WARNING: E2B_API_KEY is not set');
if (!GITHUB_USERNAME || !GITHUB_TOKEN) {
  console.warn('WARNING: GITHUB_USERNAME/GITHUB_TOKEN not set — commit_and_push tool will fail until set');
}

const APP_SECRET = process.env.APP_SECRET;
if (!APP_SECRET) console.warn('WARNING: APP_SECRET is not set — /chat and /session endpoints are unprotected');

function requireAppSecret(req, res, next) {
  if (!APP_SECRET) return next(); // not configured yet, don't lock the user out accidentally
  const provided = req.headers['x-app-secret'];
  if (provided !== APP_SECRET) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  next();
}

let aiClient = null;
function getGeminiClient() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY environment variable is required');
  }
  if (!aiClient) {
    aiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        }
      }
    });
  }
  return aiClient;
}

let groqClient = null;
function getGroqClient() {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) throw new Error('GROQ_API_KEY environment variable is required');
  if (!groqClient) {
    groqClient = new Groq({ apiKey });
  }
  return groqClient;
}

let mistralClient = null;
function getMistralClient() {
  const apiKey = process.env.MISTRAL_API_KEY;
  if (!apiKey) throw new Error('MISTRAL_API_KEY environment variable is required');
  if (!mistralClient) {
    mistralClient = new Mistral({ apiKey });
  }
  return mistralClient;
}

let cohereClient = null;
function getCohereClient() {
  const apiKey = process.env.COHERE_API_KEY;
  if (!apiKey) throw new Error('COHERE_API_KEY environment variable is required');
  if (!cohereClient) {
    cohereClient = new CohereClient({ token: apiKey });
  }
  return cohereClient;
}

let openrouterClient = null;
function getOpenRouterClient() {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) throw new Error('OPENROUTER_API_KEY environment variable is required');
  if (!openrouterClient) {
    openrouterClient = new OpenAI({ apiKey, baseURL: 'https://openrouter.ai/api/v1' });
  }
  return openrouterClient;
}

// ---- In-memory sandbox registry (per session) ----
// NOTE: Cloud Run can scale to multiple instances, and each instance has its
// own memory. For a solo/single-user project, set Cloud Run's max instances
// to 1 in the service settings so this Map stays consistent across requests.
// sandboxes: sessionId -> { sandbox, lastUsed: <timestamp> }
const sandboxes = new Map();

async function getOrCreateSandbox(sessionId) {
  let entry = sandboxes.get(sessionId);
  if (entry) {
    try {
      // cheap check that the sandbox is still alive
      await entry.sandbox.commands.run('echo ok', { timeoutMs: 5000 });
      entry.lastUsed = Date.now();
      return entry.sandbox;
    } catch {
      sandboxes.delete(sessionId);
    }
  }
  const sbx = await Sandbox.create({ apiKey: E2B_API_KEY });
  await sbx.commands.run(`mkdir -p ${PROJECT_DIR}`);
  await sbx.commands.run(`mkdir -p ${SKILLS_DIR}`);
  await sbx.files.write(`${SKILLS_DIR}/deployment.md`, `# Deployment Skill: Vercel & Cloudflare Pages

## Vercel
- Deploy with the \`deploy_to_vercel\` tool. It defaults to a **preview** deployment.
- Only pass \`production: true\` when the user explicitly asks for a production deploy — never default to production on your own judgment.
- The tool handles the auth token internally; you never need to see or reference the token value yourself.

## Cloudflare Pages (quick previews)
- Use \`deploy_preview_cloudflare\` for fast, disposable preview links, especially for static sites.
- Requires a \`projectName\` — reuse the same name across a project's redeploys so it updates the same Cloudflare Pages project rather than creating duplicates.
- If the project has a build step (e.g. Vite, React), run the build first (e.g. \`npm run build\`) and pass the output folder as \`buildDir\` (e.g. "dist" or "build").
- Always use the \`deploy_preview_cloudflare\` tool for Cloudflare deploys — it handles Node version requirements, project creation, and authentication internally. Never attempt to run \`wrangler\` or manage Node/nvm manually via \`run_command\`; doing so bypasses the token setup this tool provides and will fail.
- To delete a preview when the user asks to clean up, use the \`delete_cloudflare_preview\` tool with the project name. Never delete a project unless the user explicitly requests it.

## General
- Prefer a preview deployment first so the user can review before anything goes to production.
- Never print or echo token/environment variable values in any command output.
`);
  try {
    await sbx.commands.run(
      `git clone --depth 1 https://github.com/anthropics/skills.git /tmp/anthropic-skills-src 2>&1 && ` +
      `mkdir -p ${SKILLS_DIR}/anthropic && ` +
      `for d in frontend-design web-artifacts-builder; do ` +
      `if [ -d "/tmp/anthropic-skills-src/skills/$d" ]; then cp -r "/tmp/anthropic-skills-src/skills/$d" ${SKILLS_DIR}/anthropic/; fi; ` +
      `done && rm -rf /tmp/anthropic-skills-src`,
      { timeoutMs: 60000 }
    );
  } catch (err) {
    console.warn('Failed to fetch Anthropic example skills (non-fatal):', err.message);
  }
  try {
    await sbx.commands.run(
      `git clone --depth 1 https://github.com/supabase/agent-skills.git /tmp/supabase-skills-src 2>&1 && ` +
      `mkdir -p ${SKILLS_DIR}/supabase && ` +
      `for d in supabase postgres-best-practices; do ` +
      `if [ -d "/tmp/supabase-skills-src/skills/$d" ]; then cp -r "/tmp/supabase-skills-src/skills/$d" ${SKILLS_DIR}/supabase/; fi; ` +
      `done && rm -rf /tmp/supabase-skills-src`,
      { timeoutMs: 60000 }
    );
  } catch (err) {
    console.warn('Failed to fetch Supabase agent skills (non-fatal):', err.message);
  }
  try {
    const doc = await db.collection('agent_sessions').doc(sessionId).get();
    if (doc.exists && doc.data().repo && GITHUB_USERNAME && GITHUB_TOKEN) {
      const repo = doc.data().repo;
      const remoteUrl = `https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com/${repo}.git`;
      await sbx.commands.run(
        `git clone "${remoteUrl}" /tmp/_restore 2>&1 && cp -r /tmp/_restore/. ${PROJECT_DIR}/ && rm -rf /tmp/_restore`,
        { timeoutMs: 60000 }
      );
      console.log(`Restored project from ${repo} for session ${sessionId}`);
    }
  } catch (err) {
    console.warn('Failed to auto-restore project from GitHub (non-fatal):', err.message);
  }
  sandboxes.set(sessionId, { sandbox: sbx, lastUsed: Date.now() });
  return sbx;
}

// ---- Tool (function) definitions given to Gemini ----
const toolDeclarations = [
  {
    name: 'run_command',
    description:
      'Run a shell command inside the Linux sandbox and return stdout, stderr, and exit code. Use for installs, running scripts, tests, git commands, etc.',
    parameters: {
      type: 'object',
      properties: {
        command: { type: 'string', description: 'The shell command to execute.' },
        cwd: {
          type: 'string',
          description: 'Working directory to run the command in (optional).'
        }
      },
      required: ['command']
    }
  },
  {
    name: 'write_file',
    description: 'Create or overwrite a file with the given text content.',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Absolute or relative file path.' },
        content: { type: 'string', description: 'Full text content to write.' }
      },
      required: ['path', 'content']
    }
  },
  {
    name: 'read_file',
    description: 'Read and return the text content of a file.',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Absolute or relative file path.' }
      },
      required: ['path']
    }
  },
  {
    name: 'list_files',
    description: 'List files and directories at a given path (defaults to current project root).',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Directory path to list (optional).' }
      }
    }
  },
  {
    name: 'commit_and_push',
    description:
      'Commit all current changes in the project and push them to a GitHub repository. Use this when a task or project is complete and ready to save permanently.',
    parameters: {
      type: 'object',
      properties: {
        repo: {
          type: 'string',
          description:
            'Target repo in "owner/repo-name" format, e.g. "yourname/my-project". If the repo does not exist yet, create it on GitHub first.'
        },
        commitMessage: {
          type: 'string',
          description: 'Commit message describing the changes (optional, defaults to a generic message).'
        },
        cwd: {
          type: 'string',
          description: 'Project directory to push (optional, defaults to current directory).'
        }
      },
      required: ['repo']
    }
  },
  {
    name: 'web_search',
    description:
      'Search the web for current information, documentation, or anything not known from training data. Returns a list of relevant results with titles, URLs, and content snippets.',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'The search query.' },
        max_results: { type: 'integer', description: 'Number of results to return (optional, defaults to 5).' }
      },
      required: ['query']
    }
  },
  {
    name: 'deploy_to_vercel',
    description: 'Deploy the current project to Vercel. Defaults to a preview deployment; only pass production: true when the user explicitly asks to deploy to production. Read /home/user/skills/deployment.md first if unsure how to use this.',
    parameters: {
      type: 'object',
      properties: {
        production: { type: 'boolean', description: 'Deploy to production instead of preview (optional, defaults to false).' },
        cwd: { type: 'string', description: 'Project directory to deploy (optional, defaults to the project root).' }
      }
    }
  },
  {
    name: 'deploy_preview_cloudflare',
    description: 'Deploy the current project to Cloudflare Pages for a quick preview URL. Read /home/user/skills/deployment.md first if unsure how to use this.',
    parameters: {
      type: 'object',
      properties: {
        projectName: { type: 'string', description: 'Cloudflare Pages project name (will be created if it does not exist).' },
        buildDir: { type: 'string', description: 'Directory containing the built/static site to deploy (e.g. "dist" or "build"). Defaults to the project root if not specified.' },
        cwd: { type: 'string', description: 'Project directory to run the deploy from (optional, defaults to the project root).' }
      },
      required: ['projectName']
    }
  },
  {
    name: 'delete_cloudflare_preview',
    description: 'Permanently delete a Cloudflare Pages project and all its deployments. Only use when the user explicitly asks to delete/clean up a preview.',
    parameters: {
      type: 'object',
      properties: {
        projectName: { type: 'string', description: 'The Cloudflare Pages project name to delete.' }
      },
      required: ['projectName']
    }
  },
  {
    name: 'browse_webpage',
    description: 'Open a specific URL and return its fully-rendered text content (handles JavaScript-rendered pages). Use this when you need to read a specific page in full, as opposed to web_search which only returns search result snippets.',
    parameters: {
      type: 'object',
      properties: {
        url: { type: 'string', description: 'The full URL to open and read.' }
      },
      required: ['url']
    }
  },
  {
    name: 'take_screenshot',
    description: 'Take a screenshot of a URL (e.g. a deployed preview) and save it as an image file in the project, so you and the user can visually verify what was deployed.',
    parameters: {
      type: 'object',
      properties: {
        url: { type: 'string', description: 'The URL to screenshot.' },
        filename: { type: 'string', description: 'Filename to save as, e.g. "preview.png" (optional, defaults to "screenshot.png").' }
      },
      required: ['url']
    }
  },
  {
    name: 'clone_project',
    description: 'Clone an existing GitHub repository into the project directory to resume work on a previous project, instead of starting from scratch.',
    parameters: {
      type: 'object',
      properties: {
        repo: { type: 'string', description: 'Repo in "owner/repo-name" format.' }
      },
      required: ['repo']
    }
  },
  {
    name: 'generate_image',
    description: 'Generate an image from a text description (e.g. for a placeholder logo or hero image) and save it into the project.',
    parameters: {
      type: 'object',
      properties: {
        prompt: { type: 'string', description: 'Description of the image to generate.' },
        filename: { type: 'string', description: 'Filename to save as, e.g. "hero.png" (optional, defaults to "generated-image.png").' }
      },
      required: ['prompt']
    }
  },
  {
    name: 'patch_file',
    description: 'Edit an existing file by replacing one exact, unique snippet of text with another, without rewriting the whole file. Prefer this over write_file for any change to an existing file — it is faster and safer, especially for larger files. old_str must appear in the file exactly once; if it appears zero or multiple times, the edit is rejected so you can adjust it to be unique.',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'File path to edit.' },
        old_str: { type: 'string', description: 'The exact existing text to find and replace. Must be unique within the file — include enough surrounding context if needed.' },
        new_str: { type: 'string', description: 'The text to replace it with. Use an empty string to delete the matched text.' }
      },
      required: ['path', 'old_str', 'new_str']
    }
  }
];

// ---- Execute a single tool call against the sandbox ----
async function executeTool(sbx, name, args, sessionId) {
  try {
    switch (name) {
      case 'run_command': {
        try {
          const result = await sbx.commands.run(args.command, {
            cwd: args.cwd || PROJECT_DIR,
            timeoutMs: 120000 // 2 min per command; tune as needed
          });
          return {
            stdout: redactSecrets(result.stdout),
            stderr: redactSecrets(result.stderr),
            exitCode: result.exitCode
          };
        } catch (err) {
          return {
            stdout: redactSecrets(err.stdout),
            stderr: redactSecrets(err.stderr || err.message),
            exitCode: err.exitCode !== undefined ? err.exitCode : 1
          };
        }
      }
      case 'write_file': {
        const filePath = args.path.startsWith('/') ? args.path : `${PROJECT_DIR}/${args.path}`;
        await sbx.files.write(filePath, args.content);
        return { success: true };
      }
      case 'patch_file': {
        const filePath = args.path.startsWith('/') ? args.path : `${PROJECT_DIR}/${args.path}`;
        let content;
        try {
          content = await sbx.files.read(filePath);
        } catch (err) {
          return { error: `Could not read file: ${err.message || err}` };
        }
        const occurrences = content.split(args.old_str).length - 1;
        if (occurrences === 0) {
          return { error: 'old_str not found in file. No changes made.' };
        }
        if (occurrences > 1) {
          return { error: `old_str found ${occurrences} times in file — must be unique. Add more surrounding context to old_str and try again.` };
        }
        const newContent = content.replace(args.old_str, args.new_str);
        await sbx.files.write(filePath, newContent);
        return { success: true, path: filePath };
      }
      case 'read_file': {
        const filePath = args.path.startsWith('/') ? args.path : `${PROJECT_DIR}/${args.path}`;
        const content = await sbx.files.read(filePath);
        return { content };
      }
      case 'list_files': {
        const targetPath = args.path
          ? (args.path.startsWith('/') ? args.path : `${PROJECT_DIR}/${args.path}`)
          : PROJECT_DIR;
        const entries = await sbx.files.list(targetPath);
        return { entries };
      }
      case 'commit_and_push': {
        if (!GITHUB_USERNAME || !GITHUB_TOKEN) {
          return { error: 'GITHUB_USERNAME/GITHUB_TOKEN not configured on the server' };
        }
        const cwd = args.cwd || PROJECT_DIR;
        const msg = (args.commitMessage || 'Update from AI agent').replace(/"/g, '\\"');
        // Token is injected here on the server side only — Gemini never sees
        // the raw token, since it only ever supplies "repo" and "commitMessage".
        const remoteUrl = `https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com/${args.repo}.git`;

        const steps = [
          `git rev-parse --is-inside-work-tree || git init`,
          `git config user.email "agent@local"`,
          `git config user.name "${GITHUB_USERNAME}"`,
          `git add -A`,
          `git commit -m "${msg}" --allow-empty`,
          `git branch -M main`,
          `git remote remove origin 2>/dev/null; git remote add origin "${remoteUrl}"`,
          `git push -u origin main`
        ];

        const output = [];
        for (const step of steps) {
          let result;
          try {
            result = await sbx.commands.run(step, { cwd, timeoutMs: 60000 });
          } catch (err) {
            result = {
              stdout: err.stdout,
              stderr: err.stderr || err.message,
              exitCode: err.exitCode !== undefined ? err.exitCode : 1
            };
          }
          output.push({
            step: redactSecrets(step),
            stdout: redactSecrets(result.stdout),
            stderr: redactSecrets(result.stderr),
            exitCode: result.exitCode
          });
          if (result.exitCode !== 0 && step.includes('push')) {
            const stderrStr = String(result.stderr || '');
            if (stderrStr.includes('rejected') || stderrStr.includes('fetch first')) {
              const forceStep = `git push --force -u origin main`;
              let forceResult;
              try {
                forceResult = await sbx.commands.run(forceStep, { cwd, timeoutMs: 60000 });
              } catch (err) {
                forceResult = {
                  stdout: err.stdout,
                  stderr: err.stderr || err.message,
                  exitCode: err.exitCode !== undefined ? err.exitCode : 1
                };
              }
              output.push({
                step: redactSecrets(forceStep),
                stdout: redactSecrets(forceResult.stdout),
                stderr: redactSecrets(forceResult.stderr),
                exitCode: forceResult.exitCode
              });
              if (forceResult.exitCode !== 0) {
                return { error: 'git push failed', steps: output };
              }
            } else {
              return { error: 'git push failed', steps: output };
            }
          }
        }
        try {
          await db.collection('agent_sessions').doc(sessionId).set(
            { repo: args.repo, lastPushed: admin.firestore.FieldValue.serverTimestamp() },
            { merge: true }
          );
        } catch (err) {
          console.warn('Failed to record session->repo mapping in Firestore (non-fatal):', err.message);
        }
        return { success: true, steps: output };
      }
      case 'web_search': {
        const apiKey = process.env.TAVILY_API_KEY;
        if (!apiKey) return { error: 'TAVILY_API_KEY not configured on the server' };
        const response = await fetch('https://api.tavily.com/search', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            api_key: apiKey,
            query: args.query,
            max_results: args.max_results || 5
          })
        });
        if (!response.ok) {
          const errText = await response.text();
          return { error: `Tavily API error: ${response.status} ${errText}` };
        }
        const data = await response.json();
        return {
          results: (data.results || []).map((r) => ({ title: r.title, url: r.url, content: r.content }))
        };
      }
      case 'deploy_to_vercel': {
        const vercelToken = process.env.VERCEL_TOKEN;
        if (!vercelToken) return { error: 'VERCEL_TOKEN not configured on the server' };
        const cwd = args.cwd || PROJECT_DIR;
        const prodFlag = args.production ? '--prod' : '';
        const command = `VERCEL_TOKEN="${vercelToken}" npx vercel deploy --yes ${prodFlag}`.trim();
        try {
          const result = await sbx.commands.run(command, { cwd, timeoutMs: 120000 });
          return { stdout: redactSecrets(result.stdout), stderr: redactSecrets(result.stderr), exitCode: result.exitCode };
        } catch (err) {
          return { stdout: redactSecrets(err.stdout), stderr: redactSecrets(err.stderr || err.message), exitCode: err.exitCode !== undefined ? err.exitCode : 1 };
        }
      }
      case 'deploy_preview_cloudflare': {
        const cfToken = process.env.CLOUDFLARE_API_TOKEN;
        const cfAccountId = process.env.CLOUDFLARE_ACCOUNT_ID;
        if (!cfToken || !cfAccountId) return { error: 'CLOUDFLARE_API_TOKEN/CLOUDFLARE_ACCOUNT_ID not configured on the server' };
        const cwd = args.cwd || PROJECT_DIR;
        const deployDir = args.buildDir || '.';
        const command = `export NVM_DIR="$HOME/.nvm"; if [ ! -s "$NVM_DIR/nvm.sh" ]; then curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash > /dev/null 2>&1; fi; \\. "$NVM_DIR/nvm.sh"; nvm install 22 > /dev/null 2>&1; nvm use 22 > /dev/null 2>&1; CLOUDFLARE_API_TOKEN="${cfToken}" CLOUDFLARE_ACCOUNT_ID="${cfAccountId}" npx wrangler pages project create "${args.projectName}" --production-branch main > /dev/null 2>&1; CLOUDFLARE_API_TOKEN="${cfToken}" CLOUDFLARE_ACCOUNT_ID="${cfAccountId}" npx wrangler pages deploy ${deployDir} --project-name="${args.projectName}"`;
        try {
          const result = await sbx.commands.run(command, { cwd, timeoutMs: 240000 });
          return { stdout: redactSecrets(result.stdout), stderr: redactSecrets(result.stderr), exitCode: result.exitCode };
        } catch (err) {
          return { stdout: redactSecrets(err.stdout), stderr: redactSecrets(err.stderr || err.message), exitCode: err.exitCode !== undefined ? err.exitCode : 1 };
        }
      }
      case 'delete_cloudflare_preview': {
        const cfToken = process.env.CLOUDFLARE_API_TOKEN;
        const cfAccountId = process.env.CLOUDFLARE_ACCOUNT_ID;
        if (!cfToken || !cfAccountId) return { error: 'CLOUDFLARE_API_TOKEN/CLOUDFLARE_ACCOUNT_ID not configured on the server' };
        const command = `export NVM_DIR="$HOME/.nvm"; if [ ! -s "$NVM_DIR/nvm.sh" ]; then curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash > /dev/null 2>&1; fi; \\. "$NVM_DIR/nvm.sh"; nvm install 22 > /dev/null 2>&1; nvm use 22 > /dev/null 2>&1; CLOUDFLARE_API_TOKEN="${cfToken}" CLOUDFLARE_ACCOUNT_ID="${cfAccountId}" npx wrangler pages project delete "${args.projectName}" --force`;
        try {
          const result = await sbx.commands.run(command, { timeoutMs: 240000 });
          return { stdout: redactSecrets(result.stdout), stderr: redactSecrets(result.stderr), exitCode: result.exitCode };
        } catch (err) {
          return { stdout: redactSecrets(err.stdout), stderr: redactSecrets(err.stderr || err.message), exitCode: err.exitCode !== undefined ? err.exitCode : 1 };
        }
      }
      case 'browse_webpage': {
        const apiKey = process.env.BROWSERLESS_API_KEY;
        if (!apiKey) return { error: 'BROWSERLESS_API_KEY not configured on the server' };
        const response = await fetch(`https://production-sfo.browserless.io/content?token=${apiKey}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: args.url })
        });
        if (!response.ok) {
          const errText = await response.text();
          return { error: `Browserless API error: ${response.status} ${errText}` };
        }
        const html = await response.text();
        // Strip tags for a cleaner, more token-efficient result the model can actually read
        const textContent = html.replace(/<script[\s\S]*?<\/script>/gi, '')
                                 .replace(/<style[\s\S]*?<\/style>/gi, '')
                                 .replace(/<[^>]+>/g, ' ')
                                 .replace(/\s+/g, ' ')
                                 .trim()
                                 .slice(0, 8000);
        return { content: textContent };
      }
      case 'take_screenshot': {
        const apiKey = process.env.BROWSERLESS_API_KEY;
        if (!apiKey) return { error: 'BROWSERLESS_API_KEY not configured on the server' };
        const response = await fetch(`https://production-sfo.browserless.io/screenshot?token=${apiKey}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: args.url, options: { fullPage: true, type: 'png' } })
        });
        if (!response.ok) {
          const errText = await response.text();
          return { error: `Browserless screenshot error: ${response.status} ${errText}` };
        }
        const buffer = Buffer.from(await response.arrayBuffer());
        const filename = args.filename || 'screenshot.png';
        const filePath = `${PROJECT_DIR}/${filename}`;
        await sbx.files.write(filePath, buffer);
        return { success: true, path: filePath, sizeBytes: buffer.length };
      }
      case 'clone_project': {
        if (!GITHUB_USERNAME || !GITHUB_TOKEN) {
          return { error: 'GITHUB_USERNAME/GITHUB_TOKEN not configured on the server' };
        }
        const remoteUrl = `https://${GITHUB_USERNAME}:${GITHUB_TOKEN}@github.com/${args.repo}.git`;
        try {
          const result = await sbx.commands.run(
            `rm -rf ${PROJECT_DIR}/* ${PROJECT_DIR}/.[!.]* 2>/dev/null; git clone "${remoteUrl}" ${PROJECT_DIR} 2>&1 || git clone "${remoteUrl}" /tmp/_clone && cp -r /tmp/_clone/. ${PROJECT_DIR}/ && rm -rf /tmp/_clone`,
            { timeoutMs: 60000 }
          );
          return { stdout: redactSecrets(result.stdout), stderr: redactSecrets(result.stderr), exitCode: result.exitCode };
        } catch (err) {
          return { stdout: redactSecrets(err.stdout), stderr: redactSecrets(err.stderr || err.message), exitCode: err.exitCode !== undefined ? err.exitCode : 1 };
        }
      }
      case 'generate_image': {
        const encodedPrompt = encodeURIComponent(args.prompt);
        const response = await fetch(`https://image.pollinations.ai/prompt/${encodedPrompt}`);
        if (!response.ok) {
          return { error: `Image generation error: ${response.status}` };
        }
        const buffer = Buffer.from(await response.arrayBuffer());
        const filename = args.filename || 'generated-image.png';
        const filePath = `${PROJECT_DIR}/${filename}`;
        await sbx.files.write(filePath, buffer);
        return { success: true, path: filePath, sizeBytes: buffer.length };
      }
      default:
        return { error: `Unknown tool: ${name}` };
    }
  } catch (err) {
    return { error: redactSecrets(err.message || String(err)) };
  }
}

// ---- Tool-format adapter functions ----
function toOpenAITools(tools) {
  return tools.map((t) => ({
    type: 'function',
    function: {
      name: t.name,
      description: t.description,
      parameters: t.parameters
    }
  }));
}

function toCohereTools(tools) {
  return tools.map((t) => {
    const parameterDefinitions = {};
    const props = t.parameters?.properties || {};
    const required = new Set(t.parameters?.required || []);
    for (const [key, val] of Object.entries(props)) {
      parameterDefinitions[key] = {
        description: val.description || '',
        type: val.type || 'string',
        required: required.has(key)
      };
    }
    return {
      name: t.name,
      description: t.description,
      parameterDefinitions
    };
  });
}

// ---- History-format adapter functions ----
function toGeminiHistory(history) {
  if (!Array.isArray(history)) return [];
  return history.map((item) => ({
    role: item.role === 'assistant' || item.role === 'model' ? 'model' : 'user',
    parts: [{ text: item.content || '' }]
  }));
}

function toOpenAIHistory(history) {
  if (!Array.isArray(history)) return [];
  return history.map((item) => ({
    role: item.role === 'model' ? 'assistant' : item.role,
    content: item.content || ''
  }));
}

function toCohereHistory(history) {
  if (!Array.isArray(history)) return [];
  return history.map((item) => ({
    role: item.role === 'assistant' || item.role === 'model' ? 'CHATBOT' : 'USER',
    message: item.content || ''
  }));
}

// ---- Provider turn runners ----
async function runGeminiTurn(model, message, history, sbx, sessionId, onEvent = () => {}) {
  const ai = getGeminiClient();
  const chat = ai.chats.create({
    model,
    config: {
      systemInstruction: UI_SKILLS_INSTRUCTION,
      tools: [{ functionDeclarations: toolDeclarations }]
    },
    history: toGeminiHistory(history)
  });

  let response = await chat.sendMessage({ message });
  let loops = 0;
  const toolLog = [];

  while (loops < MAX_TOOL_LOOPS) {
    const functionCalls = response.functionCalls;
    if (!functionCalls || functionCalls.length === 0) break;

    const functionResponses = [];
    for (const call of functionCalls) {
      onEvent({ type: 'tool_start', tool: call.name, args: call.args });
      const toolResult = await executeTool(sbx, call.name, call.args || {}, sessionId);
      onEvent({ type: 'tool_end', tool: call.name, success: !toolResult.error });
      toolLog.push({ tool: call.name, args: call.args, result: toolResult });
      functionResponses.push({
        functionResponse: {
          id: call.id,
          name: call.name,
          response: toolResult
        }
      });
    }

    response = await chat.sendMessage({
      message: functionResponses
    });
    loops += 1;
  }

  const reply = response.text || '';
  return { reply, toolLog, loops };
}

async function runGroqTurn(model, message, history, sbx, sessionId, onEvent = () => {}) {
  const groq = getGroqClient();
  const messages = [
    { role: 'system', content: UI_SKILLS_INSTRUCTION },
    ...toOpenAIHistory(history),
    { role: 'user', content: message }
  ];
  let loops = 0;
  const toolLog = [];

  while (loops < MAX_TOOL_LOOPS) {
    const completion = await groq.chat.completions.create({
      model,
      messages,
      tools: toOpenAITools(toolDeclarations),
      tool_choice: 'auto'
    });

    const choice = completion.choices?.[0];
    const msg = choice?.message;
    const toolCalls = msg?.tool_calls;

    if (!toolCalls || toolCalls.length === 0) {
      return { reply: msg?.content || '', toolLog, loops };
    }

    messages.push(msg);

    for (const call of toolCalls) {
      let args = {};
      try {
        args = typeof call.function?.arguments === 'string' ? JSON.parse(call.function.arguments || '{}') : (call.function?.arguments || {});
      } catch {
        args = {};
      }
      onEvent({ type: 'tool_start', tool: call.function?.name, args });
      const toolResult = await executeTool(sbx, call.function?.name, args, sessionId);
      onEvent({ type: 'tool_end', tool: call.function?.name, success: !toolResult.error });
      toolLog.push({ tool: call.function?.name, args, result: toolResult });
      messages.push({
        role: 'tool',
        tool_call_id: call.id,
        content: JSON.stringify(toolResult)
      });
    }

    loops += 1;
  }

  const lastMsg = messages[messages.length - 1];
  return { reply: (lastMsg?.role === 'assistant' ? lastMsg.content : '') || '', toolLog, loops };
}

async function runMistralTurn(model, message, history, sbx, sessionId, onEvent = () => {}) {
  const mistral = getMistralClient();
  const messages = [
    { role: 'system', content: UI_SKILLS_INSTRUCTION },
    ...toOpenAIHistory(history),
    { role: 'user', content: message }
  ];
  let loops = 0;
  const toolLog = [];

  while (loops < MAX_TOOL_LOOPS) {
    const response = await mistral.chat.complete({
      model,
      messages,
      tools: toOpenAITools(toolDeclarations),
      toolChoice: 'auto'
    });

    const choice = response.choices?.[0];
    const msg = choice?.message;
    const toolCalls = msg?.toolCalls;

    if (!toolCalls || toolCalls.length === 0) {
      const reply = typeof msg?.content === 'string' ? msg.content : (msg?.content?.[0]?.text || '');
      return { reply, toolLog, loops };
    }

    messages.push(msg);

    for (const call of toolCalls) {
      let args = {};
      try {
        args = typeof call.function?.arguments === 'string' ? JSON.parse(call.function.arguments || '{}') : (call.function?.arguments || {});
      } catch {
        args = {};
      }
      onEvent({ type: 'tool_start', tool: call.function?.name, args });
      const toolResult = await executeTool(sbx, call.function?.name, args, sessionId);
      onEvent({ type: 'tool_end', tool: call.function?.name, success: !toolResult.error });
      toolLog.push({ tool: call.function?.name, args, result: toolResult });
      messages.push({
        role: 'tool',
        toolCallId: call.id,
        name: call.function?.name,
        content: JSON.stringify(toolResult)
      });
    }

    loops += 1;
  }

  const lastMsg = messages[messages.length - 1];
  const reply = typeof lastMsg?.content === 'string' ? lastMsg.content : (lastMsg?.content?.[0]?.text || '');
  return { reply, toolLog, loops };
}

async function runCohereTurn(model, message, history, sbx, sessionId, onEvent = () => {}) {
  const cohere = getCohereClient();
  let loops = 0;
  const toolLog = [];
  let toolResults = undefined;
  const chatHistory = toCohereHistory(history);
  const tools = toCohereTools(toolDeclarations);

  while (loops < MAX_TOOL_LOOPS) {
    const response = await cohere.chat({
      model,
      message,
      preamble: UI_SKILLS_INSTRUCTION,
      chatHistory,
      tools,
      toolResults
    });

    const toolCalls = response.toolCalls;
    if (!toolCalls || toolCalls.length === 0) {
      return { reply: response.text || '', toolLog, loops };
    }

    toolResults = [];
    for (const tc of toolCalls) {
      const args = tc.parameters || {};
      onEvent({ type: 'tool_start', tool: tc.name, args });
      const toolResult = await executeTool(sbx, tc.name, args, sessionId);
      onEvent({ type: 'tool_end', tool: tc.name, success: !toolResult.error });
      toolLog.push({ tool: tc.name, args, result: toolResult });
      toolResults.push({
        call: tc,
        outputs: [toolResult]
      });
    }

    loops += 1;
  }

  return { reply: '', toolLog, loops };
}

async function runOpenRouterTurn(model, message, history, sbx, sessionId, onEvent = () => {}) {
  const openrouter = getOpenRouterClient();
  const messages = [
    { role: 'system', content: UI_SKILLS_INSTRUCTION },
    ...toOpenAIHistory(history),
    { role: 'user', content: message }
  ];
  let loops = 0;
  const toolLog = [];

  while (loops < MAX_TOOL_LOOPS) {
    const completion = await openrouter.chat.completions.create({
      model,
      messages,
      tools: toOpenAITools(toolDeclarations),
      tool_choice: 'auto'
    });

    const choice = completion.choices?.[0];
    const msg = choice?.message;
    const toolCalls = msg?.tool_calls;

    if (!toolCalls || toolCalls.length === 0) {
      return { reply: msg?.content || '', toolLog, loops };
    }

    messages.push(msg);

    for (const call of toolCalls) {
      let args = {};
      try {
        args = typeof call.function?.arguments === 'string' ? JSON.parse(call.function.arguments || '{}') : (call.function?.arguments || {});
      } catch {
        args = {};
      }
      onEvent({ type: 'tool_start', tool: call.function?.name, args });
      const toolResult = await executeTool(sbx, call.function?.name, args, sessionId);
      onEvent({ type: 'tool_end', tool: call.function?.name, success: !toolResult.error });
      toolLog.push({ tool: call.function?.name, args, result: toolResult });
      messages.push({
        role: 'tool',
        tool_call_id: call.id,
        content: JSON.stringify(toolResult)
      });
    }

    loops += 1;
  }

  const lastMsg = messages[messages.length - 1];
  return { reply: (lastMsg?.role === 'assistant' ? lastMsg.content : '') || '', toolLog, loops };
}

const PROVIDER_RUNNERS = {
  gemini: runGeminiTurn,
  groq: runGroqTurn,
  mistral: runMistralTurn,
  cohere: runCohereTurn,
  openrouter: runOpenRouterTurn
};

// ---- Chat runner with multi-provider fallback ----
async function runChatWithFallback(message, history, sbx, sessionId, onEvent = () => {}) {
  let lastError = null;

  for (const { provider, model } of FALLBACK_CHAIN) {
    const runner = PROVIDER_RUNNERS[provider];
    if (!runner) {
      console.warn(`No runner defined for provider: ${provider}`);
      continue;
    }

    try {
      onEvent({ type: 'model_selected', provider, model });
      const { reply, toolLog, loops } = await runner(model, message, history, sbx, sessionId, onEvent);
      return {
        reply,
        toolLog,
        loopsUsed: loops,
        hitLoopCap: loops >= MAX_TOOL_LOOPS,
        modelUsed: `${provider}:${model}`
      };
    } catch (err) {
      const errMsg = err?.message || String(err);
      onEvent({ type: 'model_failed', provider, model, error: errMsg });
      console.warn(`Provider ${provider} (${model}) failed: ${errMsg}. Trying next in fallback chain...`);
      lastError = err;
    }
  }

  throw new Error(`All fallback providers and models exhausted. Last error: ${lastError?.message || 'Unknown error'}`);
}

function extractVisitedUrls(toolLog) {
  const urls = new Set();
  for (const entry of toolLog) {
    if (entry.tool === 'browse_webpage' || entry.tool === 'take_screenshot') {
      if (entry.args?.url) urls.add(entry.args.url);
    }
    if (entry.tool === 'web_search' && entry.result?.results) {
      for (const r of entry.result.results) {
        if (r.url) urls.add(r.url);
      }
    }
  }
  return Array.from(urls);
}

// ---- Main chat endpoint: runs the multi-provider <-> E2B tool-calling loop ----
app.post('/chat', requireAppSecret, async (req, res) => {
  const { message, sessionId, history } = req.body;

  if (!message || !sessionId) {
    return res.status(400).json({ error: 'message and sessionId are required' });
  }

  try {
    const sbx = await getOrCreateSandbox(sessionId);
    const result = await runChatWithFallback(message, history, sbx, sessionId);
    res.json({
      ...result,
      visitedUrls: extractVisitedUrls(result.toolLog)
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'Internal error' });
  }
});

app.post('/chat/stream', requireAppSecret, async (req, res) => {
  const { message, sessionId, history } = req.body;
  if (!message || !sessionId) {
    return res.status(400).json({ error: 'message and sessionId are required' });
  }

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
  });

  const send = (event) => {
    res.write(`data: ${JSON.stringify(event)}\n\n`);
  };

  try {
    const sbx = await getOrCreateSandbox(sessionId);
    const result = await runChatWithFallback(message, history, sbx, sessionId, send);
    send({
      type: 'done',
      reply: result.reply,
      toolLog: result.toolLog,
      loopsUsed: result.loopsUsed,
      hitLoopCap: result.hitLoopCap,
      modelUsed: result.modelUsed,
      visitedUrls: extractVisitedUrls(result.toolLog)
    });
  } catch (err) {
    send({ type: 'error', error: err.message || 'Internal error' });
  } finally {
    res.end();
  }
});

// ---- End a session explicitly (e.g. after pushing to GitHub) ----
app.post('/session/:sessionId/close', requireAppSecret, async (req, res) => {
  const { sessionId } = req.params;
  const entry = sandboxes.get(sessionId);
  if (entry) {
    try {
      await entry.sandbox.kill();
    } catch (err) {
      console.warn('Error killing sandbox:', err.message);
    }
    sandboxes.delete(sessionId);
  }
  res.json({ closed: true });
});

app.get('/health', (req, res) => res.json({ status: 'ok' }));

app.get('/', (req, res) => {
  res.json({
    status: 'ok',
    service: 'AI Coding Agent Orchestrator',
    endpoints: {
      health: 'GET /health',
      chat: 'POST /chat',
      chatStream: 'POST /chat/stream',
      closeSession: 'POST /session/:sessionId/close'
    }
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Orchestrator listening on port ${PORT}`);
});

const IDLE_TIMEOUT_MS = 20 * 60 * 1000; // 20 minutes

async function cleanupIdleSandboxes() {
  const now = Date.now();
  for (const [sessionId, entry] of sandboxes.entries()) {
    if (now - entry.lastUsed > IDLE_TIMEOUT_MS) {
      try {
        await entry.sandbox.kill();
      } catch (err) {
        console.warn(`Error killing idle sandbox ${sessionId}:`, err.message);
      }
      sandboxes.delete(sessionId);
      console.log(`Closed idle sandbox for session ${sessionId}`);
    }
  }
}

setInterval(cleanupIdleSandboxes, 5 * 60 * 1000); // check every 5 minutes
