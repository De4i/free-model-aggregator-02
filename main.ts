// free-model-aggregator-02: one server, two client wires, and a real free model behind every answer.
//
// WHAT THIS ENTRY DOES DIFFERENTLY FROM THE STARTER
//   The starter takes the first free model the brief lists that can meet a request, dials it once, and
//   loses the prompt when that call is refused. This entry plans instead:
//     1. the free models are ranked by the brief's published rating, strongest first, so a prompt is
//        served by the strongest free model its requirements admit — that is the quality column, six
//        tenths of the score, and it is the only column this contest leaves room to win;
//     2. a call that is refused walks on to the next candidate instead of ending the prompt, and the
//        refusal is reported on the answer's `x-makefun-upstream-failures` trail so the failover
//        column can see a refusal this entry recovered from;
//     3. a model that refused once is spent for the rest of the trial, so one rate-limited model never
//        takes the prompts that follow it down with it;
//     4. the routing key is the prompt's declared requirements, read once per prompt in the client's
//        arrival order, never re-derived from the answer.
//
// WHAT THE HOUSE CHECKS
//   Every answer's `model` names the model this entry really dialled: the id goes to the gateway in
//   the request and comes back on every answer and every streamed chunk, because the house scores the
//   model its own gateway record shows answering. A prompt is credited only when that record holds a
//   successful call to that model and the provider's own body carries the prompt's token. The catalog
//   is the fourteen dialable free ids, once each, in the register body and in both GET /v1/models
//   answers. Nothing here reads the wall clock or an unseeded random: a replay of the recorded answers
//   leaves the same state export, byte for byte.
//
// FILES
//   main.ts             this server: the routes, the register, the walk over the candidates, the poll.
//   lib/house-brief.ts  the brief's two documents and the catalog built from them.
//   lib/wire.ts         Anthropic <-> Chat Completions translation and the SSE writers.
//   lib/planner.ts      the ranking, the spent-model bookkeeping and the failure trail.
//   LICENSE             MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  isRecord,
  readBrief,
  type CatalogReport,
  type Json,
  type Requirement,
} from './lib/house-brief.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import { Plan, refusalCode, trailHeader, type Refusal } from './lib/planner.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this entry reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The header a recovered refusal is reported on. */
const FAILURES_HEADER = 'x-makefun-upstream-failures';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
const plan = new Plan(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** The gateway's Chat Completions route, as the OpenAI SDK would spell it for the same base URL. */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    out.push({ id: model.id, provider: model.provider, display_name: model.display_name });
  }
  return out;
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, for a brief that holds no row for this prompt. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** The body this entry dials for an OpenAI-shaped request: the client's own, with the chosen model. */
function openAIOutgoing(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  // The usage chunk is the only place a streamed Chat Completions answer carries its token counts, and
  // the wire column wants a whole input count and an output count of at least one on every answer.
  if (body.stream === true) out.stream_options = { include_usage: true };
  return out;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extra: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(text),
    ...extra,
  });
  res.end(text);
}

function sendError(
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  status: number,
  type: string,
  message: string,
  extra: Record<string, string> = {},
): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), extra);
  else sendJson(res, status, openAIError(message, type, null), extra);
}

/** The extra headers an answer carries: the failure trail, when this prompt had one. */
function trailHeaders(refusals: readonly Refusal[]): Record<string, string> {
  const trail = trailHeader(refusals);
  return trail === null ? {} : { [FAILURES_HEADER]: trail };
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one call to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify(body) });
}

/**
 * Serve one prompt: walk the ranked candidates that meet its requirements, dial each in turn, and hand
 * the first answer back in the client's own wire — streamed or not as the client asked. A refused call
 * is recorded on the trail and the walk goes on; only an empty candidate list ends the prompt.
 */
async function serve(
  req: IncomingMessage,
  res: ServerResponse,
  wire: 'openai' | 'anthropic',
  body: Json,
): Promise<void> {
  const requires = requirementsFor(body);
  const candidates = plan.candidatesFor(requires);
  if (candidates.length === 0) {
    return sendError(res, wire, 503, wire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model in the brief meets this request');
  }
  const refusals: Refusal[] = [];
  for (const model of candidates) {
    if (refusals.length >= 8) break;
    const outgoing = wire === 'anthropic' ? anthropicToOpenAI(body, model.id) : openAIOutgoing(body, model.id);
    let upstream: Response;
    try {
      upstream = await dial(model.id, outgoing);
    } catch (err) {
      // The call never reached an answer: this model is not usable for this trial.
      refusals.push({ status: 0, code: `unreachable: ${(err as Error).message}`.slice(0, 120), model: model.id });
      plan.retireOnError(model.id);
      continue;
    }
    if (!upstream.ok || upstream.body === null) {
      const text = await upstream.text().catch(() => '');
      refusals.push({ status: upstream.status, code: refusalCode(text), model: model.id });
      plan.retire(model.id, upstream.status);
      log(`refused ${model.id} with HTTP ${upstream.status}; ${plan.spentCount} model(s) spent`);
      continue;
    }
    const extra = trailHeaders(refusals);
    if (body.stream === true) {
      res.writeHead(200, {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
        ...extra,
      });
      try {
        if (wire === 'anthropic') {
          for await (const event of anthropicEventsFromOpenAI(upstream.body, model.id)) res.write(event);
        } else {
          for await (const frame of openAIStreamWithModel(upstream.body, model.id)) res.write(frame);
        }
      } finally {
        res.end();
      }
      return;
    }
    const answer: unknown = await upstream.json().catch(() => null);
    if (answer === null) {
      // A 200 whose body is not JSON: nothing to relay, so the next candidate takes the prompt.
      refusals.push({ status: 200, code: 'the gateway answered 200 with a body that is not JSON', model: model.id });
      plan.retireOnError(model.id);
      continue;
    }
    const relay = isRecord(answer) ? answer : {};
    if (wire === 'anthropic') sendJson(res, 200, openAIToAnthropic(relay, model.id), extra);
    else sendJson(res, 200, openAIJsonWithModel(relay, model.id), extra);
    return;
  }
  // Every candidate refused: the prompt is lost, and the trail says what it cost.
  const extra = trailHeaders(refusals);
  const last = refusals[refusals.length - 1];
  const detail = last === undefined ? 'every free model that meets this request refused it' : `the last candidate answered HTTP ${last.status}: ${last.code}`;
  sendError(res, wire, 502, wire === 'anthropic' ? 'api_error' : 'upstream_error', detail, extra);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(req, res, 'openai', body);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    await serve(req, res, 'anthropic', body);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  log(`ranking: ${plan.ranking.slice(0, 5).map((m) => `${m.id}=${m.rating_milli}`).join(', ')}`);
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
