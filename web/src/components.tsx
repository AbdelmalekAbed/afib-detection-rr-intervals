import type { ReactNode } from "react";
import { AlertCircle, ArrowUpRight, LoaderCircle, Info } from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { number } from "./api";
import type { Dataset, Model, Policy } from "./types";

export function Panel({
  title,
  subtitle,
  action,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}
export function Notice({ children }: { children: ReactNode }) {
  return (
    <div className="notice">
      <Info size={17} aria-hidden="true" />
      <div>{children}</div>
    </div>
  );
}
export function Loading({
  label = "Chargement des résultats validés…",
}: {
  label?: string;
}) {
  return (
    <div role="status" className="loading">
      <LoaderCircle className="spin" size={23} />
      <span>{label}</span>
    </div>
  );
}
export function ErrorBox({
  message,
  retry,
}: {
  message: string;
  retry?: () => void;
}) {
  return (
    <div role="alert" className="error">
      <AlertCircle size={22} />
      <div>
        <strong>Les données ne sont pas disponibles</strong>
        <p>{message}</p>
        {retry && (
          <button className="button small" onClick={retry}>
            Réessayer
          </button>
        )}
      </div>
    </div>
  );
}
export function MetricCard({
  label,
  value,
  detail,
  accent = false,
  icon,
}: {
  label: string;
  value: string;
  detail: string;
  accent?: boolean;
  icon?: ReactNode;
}) {
  return (
    <article className={`metric-card ${accent ? "accent" : ""}`}>
      <div className="metric-label">
        {label}
        {icon || <ArrowUpRight size={17} />}
      </div>
      <div className="metric-value">{value}</div>
      <div className="metric-detail">{detail}</div>
    </article>
  );
}
export function Filters({
  dataset,
  setDataset,
  model,
  setModel,
  policy,
  setPolicy,
}: {
  dataset: Dataset;
  setDataset: (v: Dataset) => void;
  model?: Model;
  setModel?: (v: Model) => void;
  policy: Policy;
  setPolicy: (v: Policy) => void;
}) {
  return (
    <div className="filters">
      <label>
        Base de données
        <select
          value={dataset}
          onChange={(e) => setDataset(e.target.value as Dataset)}
        >
          <option value="afdb">AFDB · validation interne</option>
          <option value="ltafdb">LTAFDB · transfert externe</option>
        </select>
      </label>
      {setModel && (
        <label>
          Modèle
          <select
            value={model}
            onChange={(e) => setModel(e.target.value as Model)}
          >
            <option value="cnn">CNN-LSTM</option>
            <option value="hrv">HRV · Gradient Boosting</option>
          </select>
        </label>
      )}
      <label>
        Règle de décision
        <select
          value={policy}
          onChange={(e) => setPolicy(e.target.value as Policy)}
        >
          <option value="fixed">Seuil fixe · 0,50</option>
          <option value="calibrated">
            {dataset === "afdb"
              ? "Seuils internes par pli"
              : "Vote · seuils AFDB gelés"}
          </option>
        </select>
      </label>
    </div>
  );
}
export function RRChart({
  rr,
  height = 180,
}: {
  rr: number[];
  height?: number;
}) {
  return (
    <div
      className="chart"
      role="img"
      aria-label="Série de 60 intervalles RR en secondes"
      style={{ height }}
    >
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={rr.map((rr, i) => ({ beat: i + 1, rr }))}
          margin={{ top: 14, right: 16, left: -18, bottom: 2 }}
        >
          <defs>
            <linearGradient id="rr-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#467e68" stopOpacity={0.18} />
              <stop offset="100%" stopColor="#467e68" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid vertical={false} stroke="#e9ece7" />
          <XAxis
            dataKey="beat"
            tickLine={false}
            axisLine={false}
            minTickGap={28}
          />
          <YAxis
            domain={[0.3, 2]}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => `${v}s`}
          />
          <Tooltip
            formatter={(value) => [
              `${number(Number(value))} s`,
              "Intervalle RR",
            ]}
            labelFormatter={(v) => `Intervalle ${v}`}
          />
          <Area
            type="linear"
            dataKey="rr"
            stroke="#467e68"
            strokeWidth={2}
            fill="url(#rr-fill)"
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
export function DecisionNote({
  dataset,
  policy,
}: {
  dataset: Dataset;
  policy: Policy;
}) {
  return (
    <p className="caption">
      {dataset === "afdb"
        ? "Chaque patient est prédit par le modèle du pli qui ne l’a pas vu à l’entraînement."
        : "Ensemble de cinq modèles entraînés sur AFDB, sans adaptation sur LTAFDB."}{" "}
      {policy === "fixed"
        ? "Décision positive lorsque le score atteint 0,50."
        : dataset === "afdb"
          ? "Le seuil propre au pli a été choisi sur ses seuls patients d’entraînement."
          : "Décision positive avec au moins 3 votes sur 5 ; chaque modèle utilise son seuil interne AFDB gelé."}
    </p>
  );
}
