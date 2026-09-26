// Shared error types + shape validation used by every backend.
export class ShapeMismatchError extends Error {
  constructor(expected, actual) {
    super(`张量形状不匹配: 期望 [${expected}], 实际 [${actual}]`);
    this.name = 'ShapeMismatchError';
    this.expected = expected;
    this.actual = actual;
  }
}

export class BackendUnavailableError extends Error {
  constructor(message) {
    super(message);
    this.name = 'BackendUnavailableError';
  }
}

export class UnsupportedOpError extends Error {
  constructor(op, detail = '') {
    super(`算子不受支持: ${op}${detail ? ` (${detail})` : ''}`);
    this.name = 'UnsupportedOpError';
    this.op = op;
  }
}

export class InferenceError extends Error {
  constructor(message, cause) {
    super(message);
    this.name = 'InferenceError';
    this.cause = cause;
  }
}

export function assertShape(actual, expected, label = 'input') {
  const ok = Array.isArray(actual) &&
    actual.length === expected.length &&
    actual.every((d, i) => d === expected[i]);
  if (!ok) {
    throw new ShapeMismatchError(expected, actual ?? 'undefined');
  }
  return true;
}
