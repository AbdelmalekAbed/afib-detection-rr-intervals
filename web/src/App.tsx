import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Cpu,
  Download,
  FlaskConical,
  Globe2,
  Layers3,
  LayoutDashboard,
  Menu,
  Microscope,
  SlidersHorizontal,
  X,
} from "lucide-react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { download, integer, number, percent, request, useRemote } from "./api";
import {
  DecisionNote,
  ErrorBox,
  Filters,
  Loading,
  MetricCard,
  Notice,
  Panel,
  RRChart,
} from "./components";
import type {
  Dataset,
  Model,
  Page,
  Patient,
  Policy,
  Prediction,
  Summary,
  Trace,
  WindowData,
} from "./types";

const pages: {
  id: Page;
  title: string;
  description: string;
  icon: typeof Activity;
}[] = [
  {
    id: "overview",
    title: "Vue d’ensemble",
    description: "Du signal RR aux résultats validés.",
    icon: LayoutDashboard,
  },
  {
    id: "patients",
    title: "Exploration des patients",
    description: "Observer les décisions, une fenêtre à la fois.",
    icon: Activity,
  },
  {
    id: "models",
    title: "Comparaison des modèles",
    description: "Deux approches, les mêmes patients.",
    icon: Layers3,
  },
  {
    id: "compression",
    title: "Compression & déploiement",
    description: "Comprendre le compromis entre qualité, taille et vitesse.",
    icon: Cpu,
  },
  {
    id: "transfer",
    title: "Généralisation externe",
    description: "Que se passe-t-il sur une base jamais vue ?",
    icon: Globe2,
  },
  {
    id: "lab",
    title: "Laboratoire RR",
    description: "Explorer la réponse du modèle sur 60 intervalles RR.",
    icon: FlaskConical,
  },
];
const navLabels = [
  "Vue d’ensemble",
  "Patients",
  "Modèles",
  "Compression",
  "Généralisation",
  "Laboratoire RR",
];
const modelLabels: Record<Model, string> = {
  cnn: "CNN-LSTM",
  hrv: "HRV · Gradient Boosting",
};
const getPage = (): Page =>
  pages.find((p) => p.id === window.location.hash.slice(1))?.id || "overview";

function App() {
  const [page, setPage] = useState<Page>(getPage);
  const [menu, setMenu] = useState(false);
  const [protocol, setProtocol] = useState(false);
  const protocolTrigger = useRef<HTMLElement | null>(null);
  const openProtocol = () => {
    protocolTrigger.current = document.activeElement as HTMLElement;
    setProtocol(true);
  };
  const remote = useRemote<Summary>("/api/summary");
  useEffect(() => {
    const listener = () => {
      setPage(getPage());
      setMenu(false);
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", listener);
    return () => window.removeEventListener("hashchange", listener);
  }, []);
  useEffect(() => {
    if (!protocol) return;
    const close = (e: KeyboardEvent) => {
      if (e.key === "Escape") setProtocol(false);
      if (e.key === "Tab") {
        const buttons = document.querySelectorAll<HTMLButtonElement>(
          "[role=dialog] button",
        );
        if (buttons.length) {
          e.preventDefault();
          buttons[0].focus();
        }
      }
    };
    window.addEventListener("keydown", close);
    return () => {
      window.removeEventListener("keydown", close);
      protocolTrigger.current?.focus();
    };
  }, [protocol]);
  const current = pages.find((p) => p.id === page)!;
  return (
    <div className="app-shell">
      {menu && (
        <button
          aria-label="Fermer le menu"
          className="nav-backdrop"
          onClick={() => setMenu(false)}
        />
      )}
      <aside className={`sidebar ${menu ? "open" : ""}`}>
        <a className="brand" href="#overview">
          <span className="brand-mark">
            <Activity size={25} />
          </span>
          <span>
            AFib<span className="brand-light">Lab</span>
            <small>ESPACE DE RECHERCHE</small>
          </span>
        </a>
        <div className="nav-section-label">EXPLORER LE PROJET</div>
        <nav aria-label="Navigation principale">
          {pages.map((p, i) => (
            <a
              key={p.id}
              href={`#${p.id}`}
              className={page === p.id ? "active" : ""}
              aria-current={page === p.id ? "page" : undefined}
            >
              <p.icon size={19} />
              <span>{navLabels[i]}</span>
              {page === p.id && <span className="nav-dot" />}
            </a>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="research-card">
            <Microscope size={22} />
            <strong>La recherche, en pratique.</strong>
            <p>Explorer les résultats et comprendre les limites du modèle.</p>
            <button onClick={openProtocol}>
              Lire le protocole <ArrowUpRight size={15} />
            </button>
          </div>
          <div className="sidebar-status">
            <span className="status-dot" />
            Candidat validé · v2
          </div>
          <small className="muted">CNN-LSTM · AFDB & LTAFDB</small>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="icon-button mobile-menu"
              aria-label="Ouvrir le menu"
              onClick={() => setMenu(true)}
            >
              <Menu size={21} />
            </button>
            <span>Projet de mémoire</span>
            <span className="slash">/</span>
            <strong>{navLabels[pages.indexOf(current)]}</strong>
          </div>
          <div className="topbar-right">
            <span className="research-badge">
              <span />
              Recherche & pédagogie
            </span>
            <button
              className="icon-button"
              aria-label="À propos du protocole"
              onClick={openProtocol}
            >
              <CircleHelp size={19} />
            </button>
            <span className="avatar" title="Espace de recherche">
              AF
            </span>
          </div>
        </header>
        <main id="main-content">
          <div className="page-heading">
            <div>
              <div className="eyebrow">AFIB DETECTION · RR INTERVALS</div>
              <h1>{current.title}</h1>
              <p>{current.description}</p>
            </div>
            <a
              className="button export-button"
              href="/api/export/summary"
              download
            >
              <Download size={16} />
              Exporter les résultats
            </a>
          </div>
          {remote.loading ? (
            <Loading />
          ) : remote.error ? (
            <ErrorBox message={remote.error} retry={remote.reload} />
          ) : (
            remote.data && (
              <>
                {page === "overview" && <Overview summary={remote.data} />}
                {page === "patients" && <Patients />}
                {page === "models" && <Models summary={remote.data} />}
                {page === "compression" && (
                  <CompressionPage summary={remote.data} />
                )}
                {page === "transfer" && <Transfer summary={remote.data} />}
                {page === "lab" && <Laboratory />}
              </>
            )
          )}
          <footer>
            <span>
              <Activity size={14} /> AFib Lab · Projet de recherche
            </span>
            <span>
              AFIB et AFL · Données publiques rétrospectives · Aucun usage
              diagnostique
            </span>
          </footer>
        </main>
      </div>
      {protocol && (
        <div className="modal-backdrop" onClick={() => setProtocol(false)}>
          <section
            className="protocol-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Protocole et limites"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              autoFocus
              className="icon-button close-modal"
              aria-label="Fermer le protocole"
              onClick={() => setProtocol(false)}
            >
              <X size={20} />
            </button>
            <span className="eyebrow">LIRE LES RÉSULTATS</span>
            <h2>Un protocole explicite.</h2>
            <p>
              AFDB : validation croisée imbriquée à cinq plis, avec séparation
              par patient. Paramètres, nombre d’époques et seuils sont
              sélectionnés sur les seuls patients d’entraînement.
            </p>
            <p>
              LTAFDB : ensemble des cinq modèles AFDB, sans entraînement ni
              ajustement de seuil sur cette base. La décision principale utilise
              la moyenne des scores au seuil de 0,50. Le vote à seuils gelés est
              une analyse secondaire.
            </p>
            <p>
              Les fenêtres contiennent 60 RR nettoyés, avec un pas de 30 RR. Le
              positif correspond à une majorité AFIB ou AFL. L’axe des fenêtres
              n’est pas une durée d’enregistrement.
            </p>
            <p>
              Le F1 moyen par patient porte sur 23/25 patients AFDB et 62/84
              patients LTAFDB. Les patients monoclasse contribuent aux métriques
              globales.
            </p>
            <Notice>
              Les scores ne sont pas des probabilités cliniques calibrées. Ces
              résultats rétrospectifs ne constituent pas une validation clinique
              prospective.
            </Notice>
          </section>
        </div>
      )}
    </div>
  );
}

function Overview({ summary: s }: { summary: Summary }) {
  const af = s.datasets.afdb.models.cnn,
    lt = s.datasets.ltafdb.models.cnn;
  const comparison = [
    {
      name: "AFDB",
      fixe: af.fixed.mean_patient_f1,
      interne: af.calibrated.mean_patient_f1,
    },
    {
      name: "LTAFDB",
      fixe: lt.fixed.mean_patient_f1,
      interne: lt.calibrated.mean_patient_f1,
    },
  ];
  return (
    <>
      <section className="hero">
        <div className="hero-copy">
          <span className="hero-tag">
            <span />
            DU SIGNAL À LA COMPRÉHENSION
          </span>
          <h2>Chaque intervalle compte.</h2>
          <p>
            Détecter les rythmes AFIB et AFL à partir des intervalles RR.
            Explorer un CNN-LSTM, ses résultats et son comportement sur des
            données jamais vues.
          </p>
          <a className="button light" href="#patients">
            Explorer les patients <ArrowRight size={17} />
          </a>
          <div className="hero-footnote">
            60 intervalles RR <span>·</span> Séparation par patient{" "}
            <span>·</span> Inférence CPU
          </div>
        </div>
        <div className="hero-visual" aria-hidden="true">
          <div className="signal-caption">
            <span>RR INTERVALS / ILLUSTRATION</span>
            <Activity size={17} />
          </div>
          <svg viewBox="0 0 430 180" className="hero-signal">
            <defs>
              <pattern
                id="grid"
                width="30"
                height="30"
                patternUnits="userSpaceOnUse"
              >
                <path
                  d="M 30 0 L 0 0 0 30"
                  fill="none"
                  stroke="#ffffff"
                  strokeOpacity=".08"
                />
              </pattern>
            </defs>
            <rect width="430" height="180" fill="url(#grid)" />
            <path
              d="M0 94L8 91L16 98L24 92L32 88L40 95L48 90L56 96L64 92L72 87L80 95L88 92L96 96L104 91L112 96L120 89L128 94L136 86L144 94L152 91L160 99L168 89L176 95L184 85L192 101L200 70L208 113L216 63L224 93L232 108L240 73L248 101L256 56L264 118L272 80L280 102L288 66L296 123L304 94L312 47L320 97L328 113L336 72L344 94L352 122L360 78L368 98L376 54L384 109L392 75L400 128L408 93L416 64L424 105L430 87"
              fill="none"
              stroke="#d2efb3"
              strokeWidth="2"
              strokeLinejoin="round"
            />
          </svg>
          <div className="signal-bottom">
            <div>
              <span>ENTRÉE DU MODÈLE</span>
              <strong>Intervalles RR</strong>
            </div>
            <span className="signal-pill">Illustration du rythme</span>
          </div>
        </div>
      </section>
      <div className="metrics-grid">
        <MetricCard
          label="F1 moyen · AFDB"
          value={number(af.calibrated.mean_patient_f1)}
          detail="Seuils internes · 23 patients évaluables"
          accent
        />
        <MetricCard
          label="F1 moyen · LTAFDB"
          value={number(lt.fixed.mean_patient_f1)}
          detail="Transfert externe · seuil fixe à 0,50"
        />
        <MetricCard
          label="F1 par fenêtre · LTAFDB"
          value={number(lt.fixed.f1)}
          detail={`${integer(s.datasets.ltafdb.n_windows)} fenêtres externes`}
        />
        <MetricCard
          label="Ensemble INT8"
          value={`${number(s.compression.int8_dynamic.ensemble_size_kib, 0)} Kio`}
          detail="5 modèles · poids sérialisés seuls"
          icon={<Cpu size={17} />}
        />
      </div>
      <div className="two-columns overview-panels">
        <Panel
          title="Des performances qui se transfèrent"
          subtitle="F1 moyen par patient · CNN-LSTM"
          action={
            <a className="text-link" href="#transfer">
              Explorer <ArrowUpRight size={15} />
            </a>
          }
        >
          <div className="chart" style={{ height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={comparison}
                barGap={8}
                margin={{ top: 15, left: -28, right: 20, bottom: 8 }}
              >
                <CartesianGrid vertical={false} stroke="#e9ece7" />
                <XAxis dataKey="name" tickLine={false} axisLine={false} />
                <YAxis domain={[0, 1]} tickLine={false} axisLine={false} />
                <Tooltip formatter={(v) => number(Number(v))} />
                <Legend iconType="circle" iconSize={8} />
                <Bar
                  name="Seuil fixe 0,50"
                  dataKey="fixe"
                  isAnimationActive={false}
                  fill="#467e68"
                  radius={[5, 5, 0, 0]}
                  maxBarSize={46}
                />
                <Bar
                  name="Seuils internes / vote gelé"
                  dataKey="interne"
                  isAnimationActive={false}
                  fill="#b3cba2"
                  radius={[5, 5, 0, 0]}
                  maxBarSize={46}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="caption">
            AFDB : prédictions hors pli. LTAFDB : ensemble externe. Les
            protocoles diffèrent ; aucun seuil n’est ajusté sur LTAFDB.
          </p>
        </Panel>
        <Panel
          title="Une évaluation, deux bases"
          subtitle="Des données publiques aux résultats reproductibles"
        >
          <div className="dataset-card">
            <span className="dataset-icon">
              <Layers3 size={20} />
            </span>
            <div>
              <strong>AFDB</strong>
              <p>Validation croisée imbriquée</p>
            </div>
            <div className="dataset-count">
              <strong>{s.datasets.afdb.n_patients}</strong>
              <small>patients</small>
            </div>
          </div>
          <div className="dataset-card">
            <span className="dataset-icon orange">
              <Globe2 size={20} />
            </span>
            <div>
              <strong>LTAFDB</strong>
              <p>Évaluation externe sans adaptation</p>
            </div>
            <div className="dataset-count">
              <strong>{s.datasets.ltafdb.n_patients}</strong>
              <small>patients</small>
            </div>
          </div>
          <div className="check-list">
            <span>
              <Check size={16} /> Sélection des paramètres sur les plis internes
            </span>
            <span>
              <Check size={16} /> Prétraitement conservant la variabilité RR
            </span>
            <span>
              <Check size={16} /> Exports ONNX vérifiés sur LTAFDB
            </span>
          </div>
          <a className="text-link" href="#models">
            Comparer avec les méthodes HRV <ArrowRight size={15} />
          </a>
        </Panel>
      </div>
      <div className="section-heading">
        <h2>Entrer dans l’expérience</h2>
        <span>Trois façons de comprendre le modèle</span>
      </div>
      <div className="explore-grid">
        {[
          {
            icon: Activity,
            name: "À l’échelle du patient",
            detail: "Confronter les scores aux annotations de référence.",
            link: "patients",
            color: "",
          },
          {
            icon: Cpu,
            name: "Plus léger, à quel prix ?",
            detail: "Comparer qualité, empreinte et latence sur CPU.",
            link: "compression",
            color: "sand",
          },
          {
            icon: FlaskConical,
            name: "Et si le rythme changeait ?",
            detail: "Modifier les RR et observer la réponse des cinq modèles.",
            link: "lab",
            color: "lavender",
          },
        ].map((card) => (
          <a className="explore-card" href={`#${card.link}`} key={card.link}>
            <span className={`explore-icon ${card.color}`}>
              <card.icon size={21} />
            </span>
            <h3>{card.name}</h3>
            <p>{card.detail}</p>
            <ArrowUpRight className="card-arrow" size={18} />
          </a>
        ))}
      </div>
      <Notice>
        Le modèle est un outil de recherche. Une irrégularité RR ne suffit pas à
        établir un diagnostic ; les classes positives de cette étude regroupent
        AFIB et AFL.
      </Notice>
    </>
  );
}

function Patients() {
  const [dataset, setDataset] = useState<Dataset>("afdb"),
    [model, setModel] = useState<Model>("cnn"),
    [policy, setPolicy] = useState<Policy>("fixed");
  const [patient, setPatient] = useState(""),
    [start, setStart] = useState(0),
    [selected, setSelected] = useState(0),
    [search, setSearch] = useState("");
  const query = `dataset=${dataset}&model=${model}&policy=${policy}`;
  const list = useRemote<Patient[]>(`/api/patients?${query}`);
  useEffect(() => {
    setPatient("");
    setStart(0);
    setSelected(0);
    setSearch("");
  }, [dataset, model, policy]);
  useEffect(() => {
    if (list.data?.length) setPatient(list.data[0].id);
  }, [list.data]);
  const trace = useRemote<Trace>(
    patient
      ? `/api/patients/${encodeURIComponent(patient)}/trace?${query}&start=${start}&limit=180`
      : null,
  );
  const rr = useRemote<WindowData>(
    patient
      ? `/api/patients/${encodeURIComponent(patient)}/window/${selected}?dataset=${dataset}`
      : null,
  );
  const choose = (id: string) => {
    setPatient(id);
    setStart(0);
    setSelected(0);
  };
  const exportVisible = () => {
    if (!trace.data) return;
    const lines = [
      "index,score,label,decision,threshold",
      ...trace.data.points.map((p) =>
        [p.index, p.score, p.label, p.decision, p.threshold].join(","),
      ),
    ];
    download(
      lines.join("\n"),
      `${dataset}_${patient}_fenetres_${start}.csv`,
      "text/csv;charset=utf-8",
    );
  };
  return (
    <>
      <Filters
        dataset={dataset}
        setDataset={(v) => {
          setPatient("");
          setDataset(v);
        }}
        model={model}
        setModel={(v) => {
          setPatient("");
          setModel(v);
        }}
        policy={policy}
        setPolicy={(v) => {
          setPatient("");
          setPolicy(v);
        }}
      />
      <DecisionNote dataset={dataset} policy={policy} />
      <div className="patient-layout">
        <Panel
          title="Patients"
          subtitle={`${list.data?.length || "…"} enregistrements · du F1 le plus faible au plus élevé`}
          className="patient-list"
        >
          <label className="search-box">
            <span className="sr-only">Rechercher un patient</span>
            <input
              placeholder="Rechercher un identifiant…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </label>
          {list.loading && <Loading label="Chargement…" />}
          {list.error && <ErrorBox message={list.error} retry={list.reload} />}
          <div className="patient-scroll">
            {list.data
              ?.filter((p) => p.id.includes(search.trim()))
              .map((p) => (
                <button
                  key={p.id}
                  className={`patient-row ${patient === p.id ? "selected" : ""}`}
                  onClick={() => choose(p.id)}
                >
                  <span className="patient-avatar">{p.id.slice(-2)}</span>
                  <span>
                    <strong>Patient {p.id}</strong>
                    <small>
                      {integer(p.n_windows)} fenêtres · {percent(p.burden)}{" "}
                      positifs
                    </small>
                  </span>
                  <span className={`f1-pill ${p.f1 === null ? "neutral" : ""}`}>
                    {p.f1 === null ? "N/A" : number(p.f1)}
                  </span>
                </button>
              ))}
            {list.data &&
              !list.data.some((p) => p.id.includes(search.trim())) && (
                <p className="caption">
                  Aucun patient ne correspond à cet identifiant.
                </p>
              )}
          </div>
          <p className="caption">
            N/A : patient monoclasse, exclu du F1 moyen par patient.
          </p>
        </Panel>
        <div className="patient-detail">
          {trace.loading && <Loading />}
          {trace.error && (
            <ErrorBox message={trace.error} retry={trace.reload} />
          )}{" "}
          {trace.data && (
            <>
              <div className="metrics-grid compact">
                <MetricCard
                  label={`Patient ${patient} · F1`}
                  value={
                    trace.data.metrics.patient_f1 === null
                      ? "N/A"
                      : number(trace.data.metrics.patient_f1)
                  }
                  detail={
                    trace.data.metrics.scorable
                      ? "Patient comportant les deux classes"
                      : "Patient monoclasse"
                  }
                />
                <MetricCard
                  label="Sensibilité"
                  value={percent(trace.data.metrics.sensitivity)}
                  detail="Fenêtres positives détectées"
                />
                <MetricCard
                  label="Spécificité"
                  value={percent(trace.data.metrics.specificity)}
                  detail="Fenêtres négatives reconnues"
                />
                <MetricCard
                  label="Charge positive"
                  value={percent(trace.data.metrics.burden)}
                  detail="Selon les annotations AFIB / AFL"
                />
              </div>
              <Panel
                title="Lecture des prédictions"
                subtitle={`${modelLabels[model]} · ${integer(trace.data.total)} fenêtres pour ce patient`}
                action={
                  <button
                    className="icon-button"
                    aria-label="Exporter les fenêtres affichées en CSV"
                    onClick={exportVisible}
                  >
                    <Download size={17} />
                  </button>
                }
              >
                <div className="legend">
                  <span>
                    <i style={{ background: "#467e68" }} />
                    Score du modèle
                  </span>
                  <span>
                    <i style={{ background: "#d0a16c" }} />
                    Référence positive
                  </span>
                  <span>
                    <i className="dashed" />
                    Décision
                  </span>
                </div>
                <div
                  className="chart trace-chart"
                  role="img"
                  aria-label="Scores et annotations sur les fenêtres affichées"
                >
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart
                      data={trace.data.points}
                      margin={{ top: 20, left: -25, right: 18, bottom: 5 }}
                      onClick={(state) => {
                        const i = Number(state?.activeTooltipIndex);
                        if (Number.isInteger(i) && trace.data?.points[i])
                          setSelected(trace.data.points[i].index);
                      }}
                    >
                      <CartesianGrid vertical={false} stroke="#e9ece7" />
                      <XAxis
                        dataKey="index"
                        tickLine={false}
                        axisLine={false}
                        minTickGap={38}
                      />
                      <YAxis
                        domain={[0, 1]}
                        tickLine={false}
                        axisLine={false}
                      />
                      <Tooltip
                        labelFormatter={(v) => `Fenêtre ${v}`}
                        formatter={(v, n) => [number(Number(v)), n]}
                      />
                      <Area
                        name="Référence positive"
                        type="stepAfter"
                        dataKey="label"
                        fill="#e3c8a9"
                        fillOpacity={0.3}
                        stroke="none"
                        isAnimationActive={false}
                      />
                      <Line
                        name="Score"
                        type="linear"
                        dataKey="score"
                        stroke="#467e68"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4 }}
                        isAnimationActive={false}
                      />
                      <Line
                        name="Décision"
                        type="stepAfter"
                        dataKey="decision"
                        stroke="#a18560"
                        strokeWidth={1}
                        strokeDasharray="4 4"
                        dot={false}
                        isAnimationActive={false}
                      />
                      {trace.data.threshold_line_is_decision_rule && (
                        <Line
                          name="Seuil"
                          type="linear"
                          dataKey="threshold"
                          stroke="#9aa49d"
                          strokeDasharray="3 4"
                          dot={false}
                          isAnimationActive={false}
                        />
                      )}
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
                <div className="trace-pagination">
                  <span>
                    Fenêtres {start}–
                    {Math.min(start + 179, trace.data.total - 1)}{" "}
                    <span className="muted">/ {trace.data.total - 1}</span>
                  </span>
                  <div>
                    <button
                      className="icon-button"
                      aria-label="Fenêtres précédentes"
                      disabled={start === 0}
                      onClick={() => setStart(Math.max(0, start - 180))}
                    >
                      <ChevronLeft size={19} />
                    </button>
                    <button
                      className="icon-button"
                      aria-label="Fenêtres suivantes"
                      disabled={start + 180 >= trace.data.total}
                      onClick={() => setStart(start + 180)}
                    >
                      <ChevronRight size={19} />
                    </button>
                  </div>
                </div>
                <p className="caption">
                  Indices des fenêtres nettoyées, sans horodatage original.
                  Cliquez sur le graphique pour inspecter une fenêtre. L’export
                  contient les fenêtres affichées.
                </p>
              </Panel>
              <Panel
                title={`Intervalles RR · fenêtre ${selected}`}
                subtitle="Entrée brute en secondes · 60 intervalles"
              >
                <label className="window-slider">
                  Choisir la fenêtre{" "}
                  <input
                    aria-label="Indice de la fenêtre RR"
                    type="range"
                    min={0}
                    max={trace.data.total - 1}
                    value={selected}
                    onChange={(e) => setSelected(Number(e.target.value))}
                  />
                  <output>{selected}</output>
                </label>
                {rr.loading && <Loading label="Lecture de la fenêtre RR…" />}
                {rr.error && (
                  <ErrorBox message={rr.error} retry={rr.reload} />
                )}{" "}
                {rr.data && (
                  <>
                    <RRChart rr={rr.data.rr} />
                    <div className="rr-meta">
                      <span className={`tag ${rr.data.label ? "orange" : ""}`}>
                        Référence :{" "}
                        {rr.data.label
                          ? "AFIB / AFL majoritaire"
                          : "Non-AFIB / AFL"}
                      </span>
                      <span className="muted">
                        RR moyen{" "}
                        {number(rr.data.rr.reduce((a, b) => a + b, 0) / 60)} s
                      </span>
                    </div>
                  </>
                )}
              </Panel>
            </>
          )}
        </div>
      </div>
      <Notice>
        Les annotations positives désignent une majorité de rythme AFIB ou AFL
        dans la fenêtre. Les patients monoclasse restent consultables ; leur F1
        est affiché N/A pour respecter la convention de l’étude.
      </Notice>
    </>
  );
}

function Models({ summary: s }: { summary: Summary }) {
  const [dataset, setDataset] = useState<Dataset>("afdb"),
    [policy, setPolicy] = useState<Policy>("fixed");
  const rows = (["cnn", "hrv"] as Model[]).map((id) => ({
    id,
    name: modelLabels[id],
    ...s.datasets[dataset].models[id][policy],
  }));
  return (
    <>
      <Filters
        dataset={dataset}
        setDataset={setDataset}
        policy={policy}
        setPolicy={setPolicy}
      />
      <DecisionNote dataset={dataset} policy={policy} />
      <Panel
        title="Deux familles de modèles"
        subtitle="Même population, même définition du F1, même règle de décision"
      >
        <div className="chart" style={{ height: 300 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={rows}
              layout="vertical"
              margin={{ left: 0, right: 35, top: 20, bottom: 20 }}
            >
              <CartesianGrid horizontal={false} stroke="#e9ece7" />
              <XAxis
                type="number"
                domain={[0, 1]}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="name"
                width={170}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip formatter={(v) => number(Number(v))} />
              <Bar
                isAnimationActive={false}
                dataKey="mean_patient_f1"
                name="F1 moyen par patient"
                radius={[0, 5, 5, 0]}
                maxBarSize={46}
              >
                {rows.map((row, i) => (
                  <Cell key={row.id} fill={i === 0 ? "#467e68" : "#b4c8a8"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Modèle</th>
                <th>F1 / patient</th>
                <th>F1 / fenêtre</th>
                <th>Sensibilité</th>
                <th>Spécificité</th>
                <th>Patients évaluables</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id}>
                  <td>
                    <strong>{row.name}</strong>
                    <small>
                      {row.id === "cnn"
                        ? "Convolutions + mémoire temporelle"
                        : "Descripteurs HRV + arbres de décision"}
                    </small>
                  </td>
                  <td className="table-highlight">
                    {number(row.mean_patient_f1)}
                  </td>
                  <td>{number(row.f1)}</td>
                  <td>{percent(row.sensitivity)}</td>
                  <td>{percent(row.specificity)}</td>
                  <td>
                    {row.n_patients_scored} / {s.datasets[dataset].n_patients}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      <div className="two-columns">
        <Panel
          title="CNN-LSTM"
          subtitle="Apprendre directement à partir des RR"
        >
          <div className="pipeline">
            <span>60 RR</span>
            <ArrowRight size={15} />
            <span>CNN</span>
            <ArrowRight size={15} />
            <span>BiLSTM</span>
            <ArrowRight size={15} />
            <span>Score</span>
          </div>
          <p>
            Les convolutions extraient des motifs locaux. Le BiLSTM représente
            les dépendances dans la séquence. Chaque pli conserve sa
            transformation et son seuil internes.
          </p>
        </Panel>
        <Panel
          title="HRV · Gradient Boosting"
          subtitle="Une référence tabulaire compétitive"
        >
          <div className="pipeline">
            <span>60 RR</span>
            <ArrowRight size={15} />
            <span>HRV</span>
            <ArrowRight size={15} />
            <span>Arbres</span>
            <ArrowRight size={15} />
            <span>Score</span>
          </div>
          <p>
            Les descripteurs résument notamment la variabilité et l’entropie des
            RR. Les hyperparamètres et seuils sont sélectionnés dans les plis
            internes.
          </p>
        </Panel>
      </div>
      <Notice>
        Ces résultats ne démontrent pas de supériorité universelle du CNN-LSTM.
        Les écarts dépendent du jeu de données et de la métrique ; les
        intervalles bootstrap appariés externes recouvrent zéro.
      </Notice>
    </>
  );
}

function CompressionPage({ summary: s }: { summary: Summary }) {
  const names: Record<string, string> = {
    fp32: "PyTorch FP32",
    int8_dynamic: "INT8 dynamique",
    onnx_fp32: "ONNX FP32",
  };
  const rows = Object.keys(names).map((id) => ({
    id,
    name: names[id],
    ...s.compression[id],
  }));
  return (
    <>
      <div className="metrics-grid three">
        <MetricCard
          label="Poids de l’ensemble INT8"
          value={`${number(s.compression.int8_dynamic.ensemble_size_kib, 0)} Kio`}
          detail="5 membres sérialisés"
          accent
        />
        <MetricCard
          label="F1 / patient · INT8"
          value={number(
            s.compression.int8_single_window.calibrated_metrics.mean_patient_f1,
          )}
          detail="AFDB · lot de 1 · seuils internes"
        />
        <MetricCard
          label="Latence ONNX · un membre"
          value={`${number(s.compression.onnx_fp32.latency_p50_ms_mean, 3)} ms`}
          detail="Moyenne des médianes · CPU · 4 threads"
        />
      </div>
      <Panel
        title="Trois formats, les mêmes modèles"
        subtitle="Mesures AFDB hors pli · seuils internes gelés"
      >
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Format</th>
                <th>F1 / patient</th>
                <th>Membre moyen</th>
                <th>Ensemble (5)</th>
                <th>Latence / membre</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id}>
                  <td>
                    <strong>{row.name}</strong>
                    <small>
                      {row.id === "int8_dynamic"
                        ? "Qualité évaluée avec lot de 512"
                        : "Scores FP32 conservés"}
                    </small>
                  </td>
                  <td>{number(row.calibrated_metrics.mean_patient_f1, 4)}</td>
                  <td>{number(row.size_kib_mean, 1)} Kio</td>
                  <td>{number(row.ensemble_size_kib, 1)} Kio</td>
                  <td>{number(row.latency_p50_ms_mean, 3)} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="caption">
          Taille des poids ou graphes uniquement : le runtime et la RAM sont
          exclus. Latence d’un membre, moyenne des médianes des cinq plis ; lot
          de 1, CPU à 4 threads, 20 échauffements et 100 répétitions. Ce ne sont
          pas des mesures sur dispositif embarqué.
        </p>
      </Panel>
      <div className="two-columns">
        <Panel
          title="Empreinte des cinq membres"
          subtitle="Kio · fichiers sérialisés"
        >
          <div className="chart" style={{ height: 245 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={rows}
                margin={{ top: 10, right: 20, left: -12, bottom: 8 }}
              >
                <CartesianGrid vertical={false} stroke="#e9ece7" />
                <XAxis dataKey="name" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} />
                <Tooltip formatter={(v) => `${number(Number(v), 1)} Kio`} />
                <Bar
                  isAnimationActive={false}
                  dataKey="ensemble_size_kib"
                  name="Ensemble"
                  radius={[5, 5, 0, 0]}
                  maxBarSize={65}
                >
                  {rows.map((r, i) => (
                    <Cell
                      key={r.id}
                      fill={["#467e68", "#c7d9bb", "#baa789"][i]}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
        <Panel
          title="Choisir selon la contrainte"
          subtitle="Des compromis mesurés"
        >
          <div className="choice">
            <Cpu size={21} />
            <div>
              <strong>Limiter la taille : INT8</strong>
              <p>
                Un ensemble plus petit, avec une qualité agrégée proche de FP32.
                INT8 est plus lent sur le CPU mesuré.
              </p>
            </div>
          </div>
          <div className="choice">
            <Activity size={21} />
            <div>
              <strong>Réduire la latence : ONNX</strong>
              <p>
                Le meilleur temps mesuré ici. L’export reproduit les scores
                externes FP32 à moins de 0,000001 près.
              </p>
            </div>
          </div>
          <div className="choice">
            <SlidersHorizontal size={21} />
            <div>
              <strong>Respecter la taille du lot</strong>
              <p>
                Les scores INT8 dépendent du contenu du lot. L’évaluation
                complète au lot de 1 confirme le maintien du F1 agrégé.
              </p>
            </div>
          </div>
        </Panel>
      </div>
      <Notice>
        Le laboratoire interactif utilise ONNX FP32 et affiche une décision
        d’ensemble. La latence du tableau concerne un membre ; elle ne
        représente pas la durée complète d’une requête web.
      </Notice>
    </>
  );
}

function Transfer({ summary: s }: { summary: Summary }) {
  const [policy, setPolicy] = useState<Policy>("fixed");
  const rows = (["afdb", "ltafdb"] as Dataset[]).map((id) => ({
    name: id.toUpperCase(),
    id,
    ...s.datasets[id].models.cnn[policy],
  }));
  return (
    <>
      <div className="transfer-banner">
        <div className="transfer-source">
          <Layers3 size={24} />
          <strong>AFDB</strong>
          <span>Entraînement & sélection interne</span>
        </div>
        <div className="transfer-arrow">
          <span>Poids et seuils gelés</span>
          <ArrowRight size={30} />
          <small>Aucune adaptation</small>
        </div>
        <div className="transfer-source">
          <Globe2 size={24} />
          <strong>LTAFDB</strong>
          <span>Évaluation externe · 84 patients</span>
        </div>
      </div>
      <div className="segment-control" aria-label="Règle de décision">
        <button
          className={policy === "fixed" ? "selected" : ""}
          onClick={() => setPolicy("fixed")}
        >
          Seuil fixe à 0,50 · analyse principale
        </button>
        <button
          className={policy === "calibrated" ? "selected" : ""}
          onClick={() => setPolicy("calibrated")}
        >
          Seuils internes / vote · analyse secondaire
        </button>
      </div>
      <div className="metrics-grid three">
        <MetricCard
          label="F1 / patient · AFDB"
          value={number(rows[0].mean_patient_f1)}
          detail="23 patients évaluables · hors pli"
        />
        <MetricCard
          label="F1 / patient · LTAFDB"
          value={number(rows[1].mean_patient_f1)}
          detail="62 patients évaluables · ensemble externe"
          accent
        />
        <MetricCard
          label="F1 / fenêtre · LTAFDB"
          value={number(rows[1].f1)}
          detail={`${integer(s.datasets.ltafdb.n_windows)} fenêtres · tous les patients`}
        />
      </div>
      <Panel
        title="Lire la généralisation à plusieurs niveaux"
        subtitle="CNN-LSTM · populations et protocoles distincts"
      >
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Base / protocole</th>
                <th>F1 / patient</th>
                <th>F1 / fenêtre</th>
                <th>Sensibilité</th>
                <th>Spécificité</th>
                <th>AUROC¹</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id}>
                  <td>
                    <strong>{row.name}</strong>
                    <small>
                      {row.id === "afdb"
                        ? "Un modèle hors pli par patient"
                        : "Ensemble de cinq modèles AFDB"}
                    </small>
                  </td>
                  <td className="table-highlight">
                    {number(row.mean_patient_f1)}
                  </td>
                  <td>{number(row.f1)}</td>
                  <td>{percent(row.sensitivity)}</td>
                  <td>{percent(row.specificity)}</td>
                  <td>{number(s.datasets[row.id].models.cnn.fixed.auroc!)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="caption">
          ¹ AUROC calculée sur les scores bruts ; elle ne change pas avec la
          règle de décision. Le F1 par fenêtre et le F1 moyen par patient
          répondent à des questions différentes.
        </p>
      </Panel>
      <div className="two-columns">
        <Panel
          title="Ce qui est vérifié"
          subtitle="Une évaluation externe complète"
        >
          <div className="check-list spacious">
            <span>
              <Check size={17} /> Les 84 patients LTAFDB contribuent aux
              métriques globales
            </span>
            <span>
              <Check size={17} /> Aucun entraînement ou réglage sur LTAFDB
            </span>
            <span>
              <Check size={17} /> FP32, INT8 et ONNX évalués sur toute la base
            </span>
            <span>
              <Check size={17} /> Les seuils du vote proviennent des seuls plis
              internes AFDB
            </span>
          </div>
        </Panel>
        <Panel
          title="Ce que ces chiffres ne suffisent pas à établir"
          subtitle="Des limites à conserver dans la soutenance"
        >
          <p>
            Une bonne performance sur deux bases publiques rétrospectives ne
            garantit pas les performances sur une nouvelle population, une
            montre ou un signal acquis en conditions réelles.
          </p>
          <p>
            Le vote à seuils gelés atteint un F1 patient supérieur ici, mais
            reste une analyse secondaire préspécifiée. Le score brut n’est pas
            une probabilité clinique.
          </p>
        </Panel>
      </div>
    </>
  );
}

function makeRR(mode: string, bpm: number, variability: number) {
  let seed = 42;
  return Array.from({ length: 60 }, (_, i) => {
    seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
    const noise = (seed / 4294967296 - 0.5) * 2;
    const amount =
      mode === "regular"
        ? 0.025
        : mode === "mixed" && i < 30
          ? 0.025
          : variability;
    return Number(
      Math.max(0.3, Math.min(2, (60 / bpm) * (1 + amount * noise))).toFixed(4),
    );
  });
}
function Laboratory() {
  const [mode, setMode] = useState("irregular"),
    [bpm, setBpm] = useState(80),
    [variability, setVariability] = useState(0.4);
  const [rr, setRR] = useState(() => makeRR("irregular", 80, 0.4)),
    [manual, setManual] = useState(false),
    [text, setText] = useState("");
  const [prediction, setPrediction] = useState<Prediction>(),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const update = (
    nextMode: string,
    nextBpm: number,
    nextVariability: number,
  ) => {
    setMode(nextMode);
    setBpm(nextBpm);
    setVariability(nextVariability);
    setRR(makeRR(nextMode, nextBpm, nextVariability));
    setPrediction(undefined);
    setError("");
  };
  const run = async () => {
    let values = rr;
    if (manual) {
      const tokens = text
        .trim()
        .split(/[\s,;]+/)
        .filter(Boolean);
      values = tokens.map(Number);
      if (
        tokens.length !== 60 ||
        values.some((v) => !Number.isFinite(v) || v < 0.3 || v > 2)
      ) {
        setError(
          "Saisir exactement 60 RR, compris entre 0.3 et 2 secondes. Utilisez le point décimal et séparez les valeurs par des espaces, virgules ou points-virgules.",
        );
        return;
      }
      setRR(values);
    }
    setBusy(true);
    setError("");
    setPrediction(undefined);
    try {
      setPrediction(
        await request<Prediction>("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ rr: values }),
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const mean = rr.reduce((a, b) => a + b, 0) / 60,
    cv = Math.sqrt(rr.reduce((a, b) => a + (b - mean) ** 2, 0) / 60) / mean;
  return (
    <>
      <Notice>
        Simulation pédagogique : les séries synthétiques ne représentent pas un
        diagnostic ni une validation supplémentaire. L’ensemble a été entraîné
        sur AFDB ; sa réponse sur un patient AFDB ne serait pas une prédiction
        hors pli.
      </Notice>
      <div className="lab-layout">
        <Panel
          title="Composer une fenêtre RR"
          subtitle="60 intervalles · entrée en secondes"
        >
          <div className="segment-control small-segments">
            <button
              disabled={busy}
              className={!manual ? "selected" : ""}
              onClick={() => {
                setManual(false);
                setPrediction(undefined);
                setError("");
              }}
            >
              Générateur
            </button>
            <button
              disabled={busy}
              className={manual ? "selected" : ""}
              onClick={() => {
                setManual(true);
                setText(rr.join(", "));
                setPrediction(undefined);
                setError("");
              }}
            >
              Saisie manuelle
            </button>
          </div>
          {!manual ? (
            <>
              <label className="form-field">
                Profil du rythme
                <select
                  disabled={busy}
                  value={mode}
                  onChange={(e) => update(e.target.value, bpm, variability)}
                >
                  <option value="regular">Régulier</option>
                  <option value="irregular">Irrégulier</option>
                  <option value="mixed">
                    Mixte · 30 réguliers puis 30 irréguliers
                  </option>
                </select>
              </label>
              <label className="range-field">
                <span>
                  Fréquence cible <strong>{bpm} bpm</strong>
                </span>
                <input
                  disabled={busy}
                  type="range"
                  min={45}
                  max={130}
                  step={1}
                  value={bpm}
                  onChange={(e) =>
                    update(mode, Number(e.target.value), variability)
                  }
                />
                <small>La fréquence réalisée dépend de la série générée.</small>
              </label>
              <label className="range-field">
                <span>
                  Amplitude des variations{" "}
                  <strong>{percent(variability)}</strong>
                </span>
                <input
                  disabled={busy || mode === "regular"}
                  type="range"
                  min={0.05}
                  max={0.7}
                  step={0.01}
                  value={variability}
                  onChange={(e) => update(mode, bpm, Number(e.target.value))}
                />
                <small>
                  Générateur déterministe · même réglage, même série.
                </small>
              </label>
            </>
          ) : (
            <label className="form-field">
              Intervalles RR en secondes
              <textarea
                disabled={busy}
                rows={9}
                value={text}
                onChange={(e) => {
                  setText(e.target.value);
                  setPrediction(undefined);
                  setError("");
                }}
                placeholder="0.80, 0.72, 0.91, …"
              />
              <small>
                60 valeurs ; point décimal ; espaces, virgules ou
                points-virgules comme séparateurs. Le graphique est actualisé à
                l’analyse.
              </small>
            </label>
          )}
          <button className="button primary full" disabled={busy} onClick={run}>
            <FlaskConical size={18} />
            {busy ? "Analyse en cours…" : "Analyser cette fenêtre"}
            <ArrowRight size={17} />
          </button>
          <p className="caption">
            Inférence réelle · ONNX FP32 · cinq modèles gelés
          </p>
        </Panel>
        <div>
          <Panel
            title="La fenêtre observée"
            subtitle="Intervalles RR bruts · pas un tracé ECG"
          >
            <RRChart rr={rr} height={230} />
            <div className="rr-stats">
              <div>
                <small>INTERVALLES</small>
                <strong>60</strong>
              </div>
              <div>
                <small>RR MOYEN</small>
                <strong>{number(mean)} s</strong>
              </div>
              <div>
                <small>VARIABILITÉ (CV)</small>
                <strong>{number(cv)}</strong>
              </div>
            </div>
          </Panel>
          {error && <ErrorBox message={error} />}{" "}
          {busy && <Loading label="Inférence des cinq modèles ONNX…" />}
          {prediction ? (
            <Panel title="Réponse de l’ensemble" subtitle={prediction.runtime}>
              <div className="prediction-result">
                <div>
                  <small>SCORE MOYEN BRUT</small>
                  <strong>{number(prediction.score)}</strong>
                </div>
                <span
                  className={`tag ${prediction.fixed_decision ? "orange" : ""}`}
                >
                  {prediction.fixed_decision
                    ? "Positif AFIB / AFL"
                    : "Négatif AFIB / AFL"}{" "}
                  · seuil 0,50
                </span>
              </div>
              <div className="score-meter">
                <span style={{ width: `${prediction.score * 100}%` }} />
              </div>
              <div className="vote-summary">
                <span>Analyse secondaire · vote à seuils AFDB</span>
                <strong>{prediction.positive_votes} / 5 positifs</strong>
              </div>
              <div className="member-list">
                {prediction.members.map((p) => (
                  <div key={p.fold}>
                    <span>Pli {p.fold + 1}</span>
                    <span>score {number(p.score)}</span>
                    <span>seuil {number(p.threshold, 2)}</span>
                    <strong className={p.decision ? "positive-text" : ""}>
                      {p.decision ? "Positif" : "Négatif"}
                    </strong>
                  </div>
                ))}
              </div>
              <p className="caption">
                {prediction.scope} Le score est une sortie du modèle, sans
                calibration clinique ni intervalle de confiance individuel.
              </p>
            </Panel>
          ) : (
            !busy && (
              <div className="empty-state">
                <FlaskConical size={27} />
                <h3>À vous d’explorer.</h3>
                <p>
                  Composez une série, puis lancez l’analyse pour observer les
                  réponses des cinq modèles.
                </p>
              </div>
            )
          )}
        </div>
      </div>
    </>
  );
}

export default App;
