# AI Coding Agent — Orchestrator

Node.js/Express backend that connects your AI Studio frontend to Gemini
(function calling) and an E2B sandbox (Linux terminal + filesystem).

## What it does

1. Frontend calls `POST /chat` with `{ message, sessionId, history }`.
2. Orchestrator sends the message to Gemini with 5 tools defined:
   `run_command`, `write_file`, `read_file`, `list_files`, `commit_and_push`.
3. When Gemini calls a tool, the orchestrator runs it inside an E2B sandbox
   tied to that `sessionId` (so the same project/session keeps its files
   between messages).
4. Loops (tool call -> execute -> feed result back to Gemini) until Gemini
   returns a normal text answer, or until `MAX_TOOL_LOOPS` (8) is hit.
5. Returns `{ reply, toolLog, loopsUsed }` to the frontend.

## Setup

```bash
npm install
cp .env.example .env
# fill in GEMINI_API_KEY and E2B_API_KEY in .env
npm start
```

Test locally:
```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"sessionId":"test1","message":"Create a Python file that prints hello world, then run it."}'
```

## Deploying via Google AI Studio -> Cloud Run

1. Push this folder to a GitHub repo.
2. In AI Studio, import the project from GitHub.
3. Set `GEMINI_API_KEY`, `E2B_API_KEY`, `GITHUB_USERNAME`, and
   `GITHUB_TOKEN` as environment variables / secrets in the Cloud Run
   service configuration (never commit `.env`).
   - `GITHUB_TOKEN`: generate at GitHub -> Settings -> Developer settings
     -> Personal access tokens -> Fine-grained tokens, scoped to just the
     repo(s) you want the agent to push to.
   - The token is only ever read server-side to build the git remote URL;
     Gemini and the frontend never see it (only the repo name and commit
     message pass through the model).
4. Deploy. Note the generated `*.run.app` URL — point your frontend's
   `/chat` calls at `<that-url>/chat`.

**Important Cloud Run setting for this project**: set **max instances to 1**
in the Cloud Run service settings. The sandbox registry is stored in memory
per-instance; if Cloud Run scales to multiple instances, a session's
sandbox could become invisible to a later request handled by a different
instance. For solo/personal use this is a one-time setting, not a code
change.

## Session lifecycle

- A sandbox is created the first time a `sessionId` is used, and reused
  for subsequent messages in that session (so files persist across turns).
- Call `POST /session/:sessionId/close` when a project is done (e.g. right
  after your agent pushes to GitHub) to free the sandbox and stay within
  E2B's concurrent sandbox limits.
- E2B's free tier caps a sandbox's lifetime — if a session runs long,
  handle a "sandbox expired" error by creating a fresh one and having the
  agent `git pull`/re-clone its own last commit to resume.

## Using commit_and_push

The agent can call this on its own when it decides a task is done — e.g.
after writing and testing code, Gemini may call
`commit_and_push({ repo: "yourname/my-project", commitMessage: "Add hello world script" })`.
The target repo must already exist on GitHub (create it first, empty is fine).

## Extending

- Add authentication (a shared secret header) on `/chat` before deploying
  publicly, since Cloud Run URLs are reachable by anyone who has the link.
