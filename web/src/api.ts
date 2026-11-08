import { useCallback, useEffect, useState } from "react";

export async function request<T>(
  url: string,
  options: RequestInit = {},
): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = `Le serveur a renvoyé une erreur (${response.status}).`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      /* Keep the HTTP status when the server response is not JSON. */
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export function useRemote<T>(url: string | null) {
  const [state, setState] = useState<{
    data?: T;
    error?: string;
    loading: boolean;
  }>({ loading: !!url });
  const [revision, setRevision] = useState(0);
  const reload = useCallback(() => setRevision((v) => v + 1), []);
  useEffect(() => {
    const controller = new AbortController();
    if (!url) {
      setState({ loading: false });
      return;
    }
    setState({ loading: true });
    request<T>(url, { signal: controller.signal })
      .then((data) => {
        if (!controller.signal.aborted) setState({ data, loading: false });
      })
      .catch((error) => {
        if (!controller.signal.aborted)
          setState({ error: error.message, loading: false });
      });
    return () => controller.abort();
  }, [url, revision]);
  return { ...state, reload };
}

export const number = (value: number, digits = 3) =>
  value.toLocaleString("fr-FR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
export const integer = (value: number) => value.toLocaleString("fr-FR");
export const percent = (value: number) => `${number(value * 100, 1)} %`;

export function download(content: string, name: string, type: string) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
