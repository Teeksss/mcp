export async function login(username: string, password: string) {
  const res = await fetch("/api/v1/auth/token", {
    method: "POST",
    body: new URLSearchParams({ username, password }),
    headers: { "Content-Type": "application/x-www-form-urlencoded" }
  });
  if (!res.ok) throw new Error("Giriş başarısız!");
  return res.json();
}