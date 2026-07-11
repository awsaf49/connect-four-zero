import { describe, expect, it } from "vitest";
import { DEMO_EVOLUTION, DEMO_REPLAYS, DEMO_RUNS } from "../src/lib/api";

describe("demo API contract", () => {
  it("ships usable fallback runs and evolution history", () => {
    expect(DEMO_RUNS.length).toBeGreaterThanOrEqual(1);
    expect(DEMO_RUNS[0]).toMatchObject({ id: "demo-bundled", status: "complete", device: "Apple M5 Max / MPS" });
    expect(DEMO_EVOLUTION.length).toBeGreaterThanOrEqual(5);
    expect(DEMO_EVOLUTION.at(-1)?.games).toBe(11162);
  });

  it("keeps replay payloads scrubbable and tied to a run", () => {
    expect(DEMO_REPLAYS.length).toBeGreaterThanOrEqual(1);
    expect(DEMO_REPLAYS.every((replay) => replay.moves.length > 0)).toBe(true);
    expect(DEMO_REPLAYS.every((replay) => replay.runId === "demo-bundled")).toBe(true);
    expect(DEMO_REPLAYS.every((replay) => replay.moves.every((column) => column >= 0 && column <= 6))).toBe(true);
  });
});
