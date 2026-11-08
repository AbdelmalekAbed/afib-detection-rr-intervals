export type Dataset = "afdb" | "ltafdb";
export type Model = "cnn" | "hrv";
export type Policy = "fixed" | "calibrated";
export type Page =
  | "overview"
  | "patients"
  | "models"
  | "compression"
  | "transfer"
  | "lab";
export interface Metrics {
  f1: number;
  mean_patient_f1: number;
  n_patients_scored: number;
  sensitivity: number;
  specificity: number;
  accuracy: number;
  auroc?: number;
  auprc?: number;
}
export interface Compression {
  metrics: Metrics;
  calibrated_metrics: Metrics;
  size_kib_mean: number;
  size_kib_min: number;
  size_kib_max: number;
  ensemble_size_kib: number;
  latency_p50_ms_mean: number;
}
export interface Summary {
  version: string;
  datasets: Record<
    Dataset,
    {
      n_windows: number;
      n_patients: number;
      models: Record<Model, { fixed: Metrics; calibrated: Metrics }>;
    }
  >;
  compression: Record<string, Compression>;
  folds: { fold: number; threshold: number; normalization: string }[];
  limitations: string[];
  latency_configuration: Record<string, unknown>;
}
export interface Patient {
  id: string;
  n_windows: number;
  burden: number;
  f1: number | null;
  scorable: boolean;
}
export interface TracePoint {
  index: number;
  score: number;
  label: number;
  decision: number;
  threshold: number;
}
export interface Trace {
  patient: string;
  total: number;
  start: number;
  points: TracePoint[];
  metrics: {
    f1: number;
    patient_f1: number | null;
    sensitivity: number;
    specificity: number;
    burden: number;
    scorable: boolean;
  };
  protocol: string;
  threshold_line_is_decision_rule: boolean;
}
export interface WindowData {
  patient: string;
  index: number;
  rr: number[];
  label: number;
}
export interface Prediction {
  score: number;
  fixed_decision: number;
  calibrated_vote: number;
  positive_votes: number;
  members: {
    fold: number;
    score: number;
    threshold: number;
    decision: number;
  }[];
  runtime: string;
  scope: string;
}
