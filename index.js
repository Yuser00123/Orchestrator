import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { Sandbox } from 'e2b';

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

const PORT = process.env.PORT || 8080;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const E2B_API_KEY = process.env.E2B_API_KEY;
const GITHUB_USERNAME = process.env.GITHUB_USERNAME;
const GITHUB_TOKEN = process.env.GITHUB_TOKEN;
const MAX_TOOL_LOOPS = 8; // safety cap so a stuck agent can't loop forever

if (!GEMINI_API_KEY) console.warn('WARNING: GEMINI_API_KEY is not set');
if (!E2B_API_KEY) console.warn('WARNING: E2B_API_KEY is not set');
if (!GITHUB_USERNAME || !GITHUB_TOKEN) {
  console.warn('WARNING: GITHUB_USERNAME/GITHUB_TOKEN not set — commit_and_push tool will fail until set');
}

const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);

// ---- In-memory sandbox registry (per session) ----
// NOTE: Cloud Run can scale to multiple instances, and each instance has its
// own memory. For a solo/single-user project, set Cloud Run's max instances
// to 1 in the service settings so this Map stays consistent across requests.
// If you ever need multi-instance scaling, swap this for a shared store
// (e.g. Firestore holding sandbox IDs) and reconnect via Sandbox.connect(id).
const sandboxes = new Map(); // sessionId -> Sandbox instance

async function getOrCreateSandbox(sessionId) {
  let sbx = sandboxes.get(sessionId);
  if (sbx) {
    try {
      // cheap check that the sandbox is still alive
      await sbx.commands.run('echo ok', { timeoutMs: 5000 });
      return sbx;
    } catch {
      sandboxes.delete(sessionId);
    }
  }
  sbx = await Sandbox.create({ apiKey: E2B_API_KEY });
  sandboxes.set(sessionId, sbx);
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
  }
];

// ---- Execute a single tool call against the sandbox ----
async function executeTool(sbx, name, args) {
  try {
    switch (name) {
      case 'run_command': {
        const result = await sbx.commands.run(args.command, {
          cwd: args.cwd,
          timeoutMs: 120000 // 2 min per command; tune as needed
        });
        return {
          stdout: result.stdout,
          stderr: result.stderr,
          exitCode: result.exitCode
        };
      }
      case 'write_file': {
        await sbx.files.write(args.path, args.content);
        return { success: true };
      }
      case 'read_file': {
        const content = await sbx.files.read(args.path);
        return { content };
      }
      case 'list_files': {
        const entries = await sbx.files.list(args.path || '.');
        return { entries };
      }
      case 'commit_and_push': {
        if (!GITHUB_USERNAME || !GITHUB_TOKEN) {
          return { error: 'GITHUB_USERNAME/GITHUB_TOKEN not configured on the server' };
        }
        const cwd = args.cwd || '.';
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
          const result = await sbx.commands.run(step, { cwd, timeoutMs: 60000 });
          // Redact the token if it ever leaks into stdout/stderr before logging/returning
          const redact = (s) => (s || '').replaceAll(GITHUB_TOKEN, '***');
          output.push({
            step: step.includes(GITHUB_TOKEN) ? step.replace(GITHUB_TOKEN, '***') : step,
            stdout: redact(result.stdout),
            stderr: redact(result.stderr),
            exitCode: result.exitCode
          });
          if (result.exitCode !== 0 && step.includes('push')) {
            return { error: 'git push failed', steps: output };
          }
        }
        return { success: true, steps: output };
      }
      default:
        return { error: `Unknown tool: ${name}` };
    }
  } catch (err) {
    return { error: err.message || String(err) };
  }
}

// ---- Main chat endpoint: runs the Gemini <-> E2B tool-calling loop ----
app.post('/chat', async (req, res) => {
  const { message, sessionId, history } = req.body;

  if (!message || !sessionId) {
    return res.status(400).json({ error: 'message and sessionId are required' });
  }

  try {
    const sbx = await getOrCreateSandbox(sessionId);

    const model = genAI.getGenerativeModel({
      model: 'gemini-2.0-flash',
      tools: [{ functionDeclarations: toolDeclarations }]
    });

    const chat = model.startChat({
      history: history || []
    });

    let response = await chat.sendMessage(message);
    let loops = 0;
    const toolLog = [];

    // Keep resolving tool calls until Gemini returns a plain text answer
    // or we hit the safety cap.
    while (loops < MAX_TOOL_LOOPS) {
      const call = response.response.functionCalls?.()?.[0];
      if (!call) break;

      const toolResult = await executeTool(sbx, call.name, call.args || {});
      toolLog.push({ tool: call.name, args: call.args, result: toolResult });

      response = await chat.sendMessage([
        {
          functionResponse: {
            name: call.name,
            response: toolResult
          }
        }
      ]);
      loops += 1;
    }

    const finalText = response.response.text();

    res.json({
      reply: finalText,
      toolLog,
      loopsUsed: loops,
      hitLoopCap: loops >= MAX_TOOL_LOOPS
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message || 'Internal error' });
  }
});

// ---- End a session explicitly (e.g. after pushing to GitHub) ----
app.post('/session/:sessionId/close', async (req, res) => {
  const { sessionId } = req.params;
  const sbx = sandboxes.get(sessionId);
  if (sbx) {
    try {
      await sbx.kill();
    } catch (err) {
      console.warn('Error killing sandbox:', err.message);
    }
    sandboxes.delete(sessionId);
  }
  res.json({ closed: true });
});

app.get('/health', (req, res) => res.json({ status: 'ok' }));

app.listen(PORT, () => {
  console.log(`Orchestrator listening on port ${PORT}`);
});
