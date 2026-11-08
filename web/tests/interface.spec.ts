import { test, expect } from "@playwright/test";

test("real results, navigation, patient trace, inference and export", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Chaque intervalle compte." }),
  ).toBeVisible();
  await expect(page.locator(".metrics-grid").first()).toContainText("0,865");
  await expect(page.locator(".recharts-bar-rectangle")).toHaveCount(4);
  await page.screenshot({
    path: "test-results/overview-desktop.png",
    fullPage: true,
  });
  await page.getByRole("link", { name: "Patients", exact: true }).click();
  await expect(page.locator(".patient-row").first()).toBeVisible();
  await expect(
    page.getByText("Lecture des prédictions", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".rr-meta")).toContainText("Référence");
  await page.screenshot({
    path: "test-results/patient-desktop.png",
    fullPage: true,
  });
  await page.getByLabel("Base de données").selectOption("ltafdb");
  await expect(page.locator(".patient-list .panel-heading")).toContainText(
    "84 enregistrements",
  );
  await page.getByLabel("Règle de décision").selectOption("calibrated");
  await expect(
    page.getByText("Lecture des prédictions", { exact: true }),
  ).toBeVisible();
  await page.getByLabel("Fenêtres suivantes").click();
  await expect(page.locator(".trace-pagination")).toContainText("180–359");
  await page.getByRole("link", { name: "Modèles", exact: true }).click();
  await expect(page.locator("table")).toContainText("0,893");
  await page.getByRole("link", { name: "Compression", exact: true }).click();
  await expect(page.locator("table")).toContainText("293,3");
  await page.getByRole("link", { name: "Généralisation", exact: true }).click();
  await expect(page.locator("table")).toContainText("0,832");
  await page
    .getByRole("button", {
      name: "Seuils internes / vote · analyse secondaire",
    })
    .click();
  await expect(page.locator("table")).toContainText("0,858");
  await page.getByRole("link", { name: "Laboratoire RR", exact: true }).click();
  await page.getByRole("button", { name: "Analyser cette fenêtre" }).click();
  await expect(
    page.getByText("Réponse de l’ensemble", { exact: true }),
  ).toBeVisible();
  await expect(page.locator(".member-list>div")).toHaveCount(5);
  await page.screenshot({
    path: "test-results/laboratory-desktop.png",
    fullPage: true,
  });
  await page
    .getByRole("button", { name: "Saisie manuelle", exact: true })
    .click();
  await page
    .getByRole("textbox", { name: "Intervalles RR en secondes" })
    .fill("0.8, 0.7");
  await page.getByRole("button", { name: "Analyser cette fenêtre" }).click();
  await expect(page.getByRole("alert")).toContainText("exactement 60 RR");
  await page
    .getByRole("textbox", { name: "Intervalles RR en secondes" })
    .fill(Array(60).fill("0.8").join(", "));
  await page.getByRole("button", { name: "Analyser cette fenêtre" }).click();
  await expect(
    page.getByText("Réponse de l’ensemble", { exact: true }),
  ).toBeVisible();
  const pending = page.waitForEvent("download");
  await page.getByRole("link", { name: "Exporter les résultats" }).click();
  expect((await pending).suggestedFilename()).toBe("afib-lab-resultats.json");
  expect(errors).toEqual([]);
});

test("mobile navigation, responsive layout and protocol dialog", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Chaque intervalle compte." }),
  ).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await page.screenshot({
    path: "test-results/overview-mobile.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Ouvrir le menu" }).click();
  await page.getByRole("link", { name: "Patients", exact: true }).click();
  await expect(
    page.getByText("Lecture des prédictions", { exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await page.getByRole("button", { name: "À propos du protocole" }).click();
  await expect(page.getByRole("dialog")).toContainText("23/25");
  await page.keyboard.press("Tab");
  await expect(
    page.getByRole("button", { name: "Fermer le protocole" }),
  ).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
});

test("API errors have a useful retry state", async ({ page }) => {
  await page.route("**/api/summary", (route) =>
    route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Catalogue indisponible pour ce test." }),
    }),
  );
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("Catalogue indisponible");
  await page.unroute("**/api/summary");
  await page.getByRole("button", { name: "Réessayer" }).click();
  await expect(
    page.getByRole("heading", { name: "Chaque intervalle compte." }),
  ).toBeVisible();
});
