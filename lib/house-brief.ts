// Entry: free-model-aggregator-02 — the brief and the catalog built from it.
//
// A trial hands the entry one file ($MAKEFUN_PROMPT_FILE): prose with two JSON documents embedded in
// it. The prose is never read. Each document is found by the schema string it carries, under this
// contest's prefix, and by the array it holds — never by a name alone, because the sibling contest
// spells the same two lists differently and a brief may carry either spelling:
//   - the model list: one row a free model, keyed by the id the gateway routes, with the house's
//     published rating in milli-points and what the model can do (tools, images, JSON, context);
//   - the prompt list: the six prompts in the order the client will send them, each with its wire,
//     whether it streams, its weight and the requirements a model must meet to serve it.
//
// The catalog this module builds is what the entry registers and what `--list` prints: one provider,
// exactly the free models the brief names, each once. The base URL is OpenRouter's public endpoint and
// never the gateway's trial-local address, because the catalog travels in the state export a replay
// compares byte for byte and the gateway's origin differs between a live pass and its replay.

import { readFileSync } from 'node:fs';

export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';
/** Every schema this contest's brief carries begins with this prefix. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';

export const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The variable name an entrant's OpenRouter key is pushed under; it never reaches the sandbox. */
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

export type Json = Record<string, unknown>;
export const isRecord = (value: unknown): value is Json =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

/** One free model as the brief lists it. `id` is the id the gateway routes. */
export interface ModelRow {
  id: string;
  provider: string;
  display_name: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs: a model without it cannot serve the prompt. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the client will send, as the brief lists it. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief, read. `error` is set when one of its documents could not be read. */
export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** What `--list` prints and what the register body carries: the same document, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

const stringOf = (value: unknown): string => (typeof value === 'string' ? value : '');
const numberOrNull = (value: unknown): number | null =>
  typeof value === 'number' && Number.isFinite(value) ? value : null;
const boolOrNull = (value: unknown): boolean | null => (typeof value === 'boolean' ? value : null);

/** The index of the brace closing the object opened at `start`, or -1; string bodies are skipped. */
function objectEnd(source: string, start: number): number {
  let depth = 0;
  let quoted = false;
  for (let i = start; i < source.length; i += 1) {
    const ch = source[i];
    if (quoted) {
      if (ch === '\\') i += 1;
      else if (ch === '"') quoted = false;
      continue;
    }
    if (ch === '"') quoted = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in the brief that carries a string `schema`, in the order it appears. */
function documentsIn(source: string): Json[] {
  const found: Json[] = [];
  for (let at = source.indexOf('{'); at !== -1; at = source.indexOf('{', at + 1)) {
    const end = objectEnd(source, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(source.slice(at, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // Not a JSON object on its own: look on.
    }
  }
  return found;
}

/**
 * The first document under this contest's prefix that carries one of `keys` as an array, with the key
 * it was found under: a brief written in the sibling contest's names still reads.
 */
function listDocument(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!stringOf(doc.schema).includes(SCHEMA_PREFIX)) continue;
    for (const key of keys) if (Array.isArray(doc[key])) return { doc, key };
  }
  return null;
}

/** One model row. A bare OpenRouter id under the provider `openrouter` is routed with its prefix. */
function modelOf(value: unknown): ModelRow | null {
  if (!isRecord(value)) return null;
  const bare = stringOf(value.id) !== '' ? stringOf(value.id) : stringOf(value.model);
  if (bare === '') return null;
  const named = stringOf(value.provider) !== '' ? stringOf(value.provider) : stringOf(value.wire);
  const routed = named === 'openrouter' && !bare.toLowerCase().startsWith('openrouter/') ? `openrouter/${bare}` : bare;
  return {
    id: routed,
    provider: named === '' ? 'openrouter' : named,
    display_name: stringOf(value.display_name) !== '' ? stringOf(value.display_name) : bare,
    rating_milli: numberOrNull(value.rating_milli) ?? 0,
    context_length: numberOrNull(value.context_length),
    tools: boolOrNull(value.tools),
    vision: boolOrNull(value.vision),
    json: boolOrNull(value.json),
    free: value.free !== false,
  };
}

/** One prompt row. */
function promptOf(value: unknown): PromptRow | null {
  if (!isRecord(value)) return null;
  const id = stringOf(value.id);
  if (id === '') return null;
  const requires = isRecord(value.requires) ? value.requires : {};
  const minContext = numberOrNull(requires.min_context);
  return {
    id,
    api: value.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: value.stream === true,
    weight: value.weight === 2 ? 2 : 1,
    requires: {
      ...(requires.tools === true ? { tools: true } : {}),
      ...(requires.vision === true ? { vision: true } : {}),
      ...(requires.json === true ? { json: true } : {}),
      ...(minContext === null ? {} : { min_context: minContext }),
    },
  };
}

/** The brief at `path`, its two documents found under this contest's schema prefix. */
export function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let source: string;
  try {
    source = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(source);
  const modelsDoc = listDocument(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models = (modelsDoc.doc[modelsDoc.key] as unknown[]).flatMap((v) => {
    const row = modelOf(v);
    return row === null ? [] : [row];
  });
  const promptsDoc = listDocument(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const prompts = (promptsDoc.doc[promptsDoc.key] as unknown[]).flatMap((v) => {
    const row = promptOf(v);
    return row === null ? [] : [row];
  });
  return { models, prompts, error: null };
}

/** The models this entry may serve: the brief's free ones, once each, in the brief's order. */
export function freeModels(brief: Brief): ModelRow[] {
  const seen = new Set<string>();
  const out: ModelRow[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    out.push(model);
  }
  return out;
}

export function capabilitiesOf(model: ModelRow): Capabilities {
  return { context_length: model.context_length, tools: model.tools, vision: model.vision, json: model.json };
}

/** Whether a model's capabilities meet one prompt's requirements. An unknown capability never does. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/** The catalog the brief's free model list makes: one provider and exactly its models. */
export function buildCatalog(brief: Brief): CatalogReport {
  const models: CatalogModel[] = freeModels(brief).map((model) => ({
    id: model.id,
    free: true,
    context_length: model.context_length,
  }));
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      { id: 'openrouter', base_url: OPENROUTER_PUBLIC_BASE, key_env: OPENROUTER_KEY_ENV, wire: 'openai', models },
    ],
  };
}
