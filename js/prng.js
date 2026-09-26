// Deterministic PRNG (mulberry32) so all backends share identical weights/inputs.
export function createPrng(seed) {
  let state = seed >>> 0;
  return function next() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randomTensor(shape, seed, scale = 0.5) {
  const size = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(size);
  const next = createPrng(seed);
  for (let i = 0; i < size; i++) {
    data[i] = (next() * 2 - 1) * scale;
  }
  return data;
}
