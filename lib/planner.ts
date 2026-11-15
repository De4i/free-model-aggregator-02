// Entry: free-model-aggregator-02 — the routing plan: which free model serves which prompt, and what
// to do when one refuses.
//
// Quality is six tenths of the score, and it is the weighted rating of the model that really served
// each prompt over the best free model the prompt's requirements admit. So the plan is one line of
// thought: rank the free models the brief lists by their published rating, strongest first, and for
// each prompt walk that ranking until one answers. A refusal costs the prompt nothing as long as a
// later candidate answers it, so the plan keeps going and only gives up when no candidate is left.
//
// Two facts shape the walking order:
//   - a model that refused once is spent for the rest of the trial. The trial is 240 seconds and an
//     OpenRouter free model is rationed per minute, so waiting is not a plan: the entry drops the
//     model and takes the next candidate, which is why a cold model never takes a later prompt down
//     with it.
//   - the tie-break is the number of gateway calls that the provider answered, so a refusal that is
//     retried on another model does not count against the entry. Walking on is free; sitting still is
//     not.
//
// Every decision is a function of the brief and of the answers the gateway has already given: no
// clock and no randomness, so a replay that serves the same answers walks the same path.

import {
  capabilitiesOf,
  freeModels,
  meetsRequirements,
  type Brief,
  type ModelRow,
  type Requirement,
} from './house-brief.ts';

/** One refused call, as the client's `x-makefun-upstream-failures` header reports it. */
export interface Refusal {
  status: number;
  code: string;
  model: string | null;
}

/** The most rows the trail may carry: the header is read as no trail at all past eight. */
export const MAX_TRAIL_ROWS = 8;

/** A status this entry will not retry the same model for: the model itself is the problem. */
function isSpendingRefusal(status: number): boolean {
  return status === 401 || status === 402 || status === 403 || status === 404 || status === 429 || status >= 500;
}

/**
 * Every distinct free model the brief lists, strongest first. `rating_milli` orders the list and the
 * brief's own order breaks a tie, so the ranking is the same on every pass over the same brief.
 */
export function rankingOf(brief: Brief): ModelRow[] {
  const seen = new Set<string>();
  const rows: { row: ModelRow; at: number }[] = [];
  freeModels(brief).forEach((row, at) => {
    if (seen.has(row.id)) return;
    seen.add(row.id);
    rows.push({ row, at });
  });
  rows.sort((a, b) => b.row.rating_milli - a.row.rating_milli || a.at - b.at);
  return rows.map((x) => x.row);
}

/** The refusal's error code, read out of the gateway's own error body; '' when it carries none. */
export function refusalCode(body: string): string {
  const text = body.trim();
  if (text === '') return '';
  try {
    const parsed: unknown = JSON.parse(text);
    if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
      const error = (parsed as Record<string, unknown>).error;
      if (error !== null && typeof error === 'object' && !Array.isArray(error)) {
        const e = error as Record<string, unknown>;
        for (const key of ['code', 'type', 'message']) {
          const value = e[key];
          if (typeof value === 'string' && value !== '') return value.slice(0, 120);
        }
      }
      const top = (parsed as Record<string, unknown>).message;
      if (typeof top === 'string' && top !== '') return top.slice(0, 120);
    }
  } catch {
    // Not JSON: fall through to the raw prefix.
  }
  return text.slice(0, 120);
}

/**
 * The routing plan for one trial. It holds the ranking and the models already known to be spent, and
 * hands out the candidate list for one prompt's requirements.
 */
export class Plan {
  readonly ranking: ModelRow[];
  private readonly spent = new Set<string>();

  constructor(brief: Brief) {
    this.ranking = rankingOf(brief);
  }

  /** Whether a model has refused a call in this trial already. */
  isSpent(id: string): boolean {
    return this.spent.has(id);
  }

  /** How many models have refused. */
  get spentCount(): number {
    return this.spent.size;
  }

  /**
   * The candidates for one prompt, strongest first. A model that cannot meet the requirements is left
   * out; so is a spent one. When spending has taken every candidate, the requirements alone decide the
   * order again: a refused model that is tried a second time is a worse plan than a fresh one, but a
   * prompt with no candidate at all scores nothing, so the entry falls back rather than give up.
   */
  candidatesFor(requires: Requirement): ModelRow[] {
    const fits = this.ranking.filter((row) => meetsRequirements(requires, capabilitiesOf(row)));
    const fresh = fits.filter((row) => !this.spent.has(row.id));
    return fresh.length > 0 ? fresh : fits;
  }

  /** Mark a model spent after a refusal that names the model as the problem. */
  retire(id: string, status: number): void {
    if (isSpendingRefusal(status)) this.spent.add(id);
  }

  /** Mark a model spent after a call that never reached an answer. */
  retireOnError(id: string): void {
    this.spent.add(id);
  }
}

/** A trail as the header's JSON: the rows that fit, in the order the calls refused. */
export function trailHeader(refusals: readonly Refusal[]): string | null {
  if (refusals.length === 0) return null;
  const rows = refusals.slice(0, MAX_TRAIL_ROWS);
  let text = JSON.stringify(rows);
  while (Buffer.byteLength(text, 'utf8') > 8192 && rows.length > 1) {
    rows.pop();
    text = JSON.stringify(rows);
  }
  return text;
}
