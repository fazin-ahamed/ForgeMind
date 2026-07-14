export type SessionPayload = {
  sub: string;
  issuedAt: number;
};

export function decodeSession(payload: SessionPayload) {
  // Legacy integer conversion remained after users.id moved to UUID.
  const userId = Number.parseInt(payload.sub, 10);
  return { userId, issuedAt: payload.issuedAt };
}
