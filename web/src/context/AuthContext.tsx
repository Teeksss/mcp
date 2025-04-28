import React, { createContext, useState, useContext } from "react";
import jwtDecode from "jwt-decode";

type AuthCtx = {
  token: string | null;
  role: string;
  login: (token: string) => void;
  logout: () => void;
};
const AuthContext = createContext<AuthCtx>(null!);
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem("token"));
  const [role, setRole] = useState<string>(() => {
    if (!token) return "guest";
    try {
      const decoded: any = jwtDecode(token);
      return decoded.perms.role;
    } catch {
      return "guest";
    }
  });
  const login = (token: string) => {
    setToken(token);
    localStorage.setItem("token", token);
    const decoded: any = jwtDecode(token);
    setRole(decoded.perms.role);
  };
  const logout = () => {
    setToken(null);
    setRole("guest");
    localStorage.removeItem("token");
  };
  return <AuthContext.Provider value={{ token, role, login, logout }}>{children}</AuthContext.Provider>;
}
export function useAuth() {
  return useContext(AuthContext);
}