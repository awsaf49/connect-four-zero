let audioContext: AudioContext | null = null;

function getAudioContext(): AudioContext | null {
  if (typeof window === "undefined" || !window.AudioContext) return null;
  audioContext ??= new window.AudioContext();
  return audioContext;
}

export function playUiSound(kind: "drop" | "win" | "select", muted: boolean): void {
  if (muted) return;
  const context = getAudioContext();
  if (!context) return;
  if (context.state === "suspended") void context.resume();

  const oscillator = context.createOscillator();
  const gain = context.createGain();
  const now = context.currentTime;
  const settings = {
    drop: { start: 220, end: 146, duration: 0.12, volume: 0.045 },
    select: { start: 330, end: 440, duration: 0.1, volume: 0.035 },
    win: { start: 392, end: 660, duration: 0.28, volume: 0.055 },
  }[kind];

  oscillator.type = kind === "win" ? "sine" : "triangle";
  oscillator.frequency.setValueAtTime(settings.start, now);
  oscillator.frequency.exponentialRampToValueAtTime(settings.end, now + settings.duration);
  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.exponentialRampToValueAtTime(settings.volume, now + 0.015);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + settings.duration);
  oscillator.connect(gain);
  gain.connect(context.destination);
  oscillator.start(now);
  oscillator.stop(now + settings.duration + 0.02);
}
