import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  timeout: 30000,
  workers: 1,
  use: {
    baseURL: process.env.AFIB_TEST_URL || "http://127.0.0.1:8000",
    headless: true,
    viewport: { width: 1440, height: 1050 },
    launchOptions: process.env.AFIB_CHROMIUM
      ? { executablePath: process.env.AFIB_CHROMIUM, args: ["--no-sandbox"] }
      : {},
  },
});
