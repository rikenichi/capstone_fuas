import React from "react";

import { useState } from "react";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL || "";

const FEATURE_LABELS = {
  NEM: "NEM",
  QUINTIL_SE4: "Quintil socioeconómico",
  COD_DEPE: "Dependencia del establecimiento",
  EDAD: "Edad",
  GENERO: "Género",
  NACIONALIDAD: "Nacionalidad",
};

const VALUE_LABELS = {
  COD_DEPE: {
    1: "Corporación Municipal",
    2: "Municipal DAEM",
    3: "Particular Subvencionado",
    4: "Particular Pagado",
    5: "Corporación de Administración Delegada",
    6: "Servicio Local de Educación Pública (SLEP)",
  },

  GENERO: {
    1: "Hombre",
    2: "Mujer",
  },

  NACIONALIDAD: {
    0: "Chileno",
    1: "Extranjero",
    2: "Nacionalizado",
  },
};

function mostrarValor(variable, valor) {
  const mapa = VALUE_LABELS[variable];
  return mapa?.[valor] ?? valor;
}

const initialForm = {
  NEM: 650,
  QUINTIL_SE4: 2,
  COD_DEPE: 3,
  EDAD: 19,
  GENERO: 2,
  NACIONALIDAD: 0,
};


function intensidadShap(impacto, factores) {
  const maxImpacto = Math.max(
    ...factores.map((f) => Math.abs(f.impacto_shap)),
    0
  );

  if (maxImpacto === 0) {
    return "Nula";
  }

  const relativo = Math.abs(impacto) / maxImpacto;

  if (relativo >= 0.66) return "Alta";
  if (relativo >= 0.33) return "Media";
  return "Baja";
}

function App() {
  const [form, setForm] = useState(initialForm);
  const [resultado, setResultado] = useState(null);
  const [explicacion, setExplicacion] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // Dashboard histórico
  const [vista, setVista] = useState("estimacion");
  const [dashboard, setDashboard] = useState(null);
  const [dashboardLoading, setDashboardLoading] = useState(false);
  const [dashboardError, setDashboardError] = useState("");
  const [dashboardYear, setDashboardYear] = useState(2025);

  const handleChange = (e) => {
    const { name, value } = e.target;

    setForm((prev) => ({
      ...prev,
      [name]: Number(value),
    }));
  };

  const analizar = async (e) => {
    e.preventDefault();

    setLoading(true);
    setError("");
    setResultado(null);
    setExplicacion(null);

    try {
      const predictResponse = await fetch(
        `${API_URL}/predict`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(form),
        }
      );

      if (!predictResponse.ok) {
        const detalle =
          await predictResponse.json();

        throw new Error(
          detalle.detail ||
          "No fue posible realizar la estimación."
        );
      }

      const predictData =
        await predictResponse.json();

      const explainResponse = await fetch(
        `${API_URL}/explain`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(form),
        }
      );

      if (!explainResponse.ok) {
        throw new Error(
          "La predicción se realizó, pero no fue posible generar la explicación."
        );
      }

      const explainData =
        await explainResponse.json();

      setResultado(predictData);
      setExplicacion(explainData);

    } catch (err) {
      setError(err.message);

    } finally {
      setLoading(false);
    }
  };

  const abrirDashboard = async () => {
    setVista("dashboard");

    if (dashboard) {
      return;
    }

    setDashboardLoading(true);
    setDashboardError("");

    try {
      const response = await fetch(
        `${API_URL}/dashboard`
      );

      if (!response.ok) {
        throw new Error(
          "No fue posible cargar el análisis histórico."
        );
      }

      const data = await response.json();

      setDashboard(data);

    } catch (err) {
      setDashboardError(err.message);

    } finally {
      setDashboardLoading(false);
    }
  };

  const resumenAnual =
    dashboard?.resumen_anual || [];

  const datosGenero =
    dashboard?.por_genero?.filter(
      (item) => item.anio === dashboardYear
    ) || [];

  const datosQuintil =
    dashboard?.por_quintil?.filter(
      (item) => item.anio === dashboardYear
    ) || [];

  const datosEdad =
    dashboard?.por_edad?.filter(
      (item) => item.anio === dashboardYear
    ) || [];

  const resumenSeleccionado =
    resumenAnual.find(
      (item) => item.anio === dashboardYear
    );

  return (
    <main className="page">

      <section className="hero">
        <div>
          <span className="tag">
            Capstone FUAS
          </span>

          <h1>
            Estimación de beneficios
            para educación superior
          </h1>

          <p>
            Herramienta analítica basada en
            patrones históricos de postulaciones
            y asignaciones FUAS.
          </p>
        </div>
      </section>

      <nav className="app-tabs">
        <button
          type="button"
          className={
            vista === "estimacion"
              ? "tab-button active"
              : "tab-button"
          }
          onClick={() => setVista("estimacion")}
        >
          Estimación individual
        </button>

        <button
          type="button"
          className={
            vista === "dashboard"
              ? "tab-button active"
              : "tab-button"
          }
          onClick={abrirDashboard}
        >
          Análisis histórico
        </button>
      </nav>

      {vista === "estimacion" && (
      <section className="workspace">

        <div className="card">

          <h2>
            Datos para la estimación
          </h2>

          <form onSubmit={analizar}>

            <div className="grid">

              <label>
                NEM

                <input
                  type="number"
                  name="NEM"
                  min="100"
                  max="700"
                  value={form.NEM}
                  onChange={handleChange}
                  required
                />
              </label>

              <label>
                Quintil socioeconómico

                <select
                  name="QUINTIL_SE4"
                  value={form.QUINTIL_SE4}
                  onChange={handleChange}
                >
                  <option value="1">
                    Quintil 1
                  </option>

                  <option value="2">
                    Quintil 2
                  </option>

                  <option value="3">
                    Quintil 3
                  </option>

                  <option value="4">
                    Quintil 4
                  </option>

                  <option value="5">
                    Quintil 5
                  </option>
                </select>
              </label>

              <label>
                Dependencia del establecimiento

                <select
                  name="COD_DEPE"
                  value={form.COD_DEPE}
                  onChange={handleChange}
                >
                  <option value="1">
                    Corporación Municipal
                  </option>

                  <option value="2">
                    Municipal DAEM
                  </option>

                  <option value="3">
                    Particular Subvencionado
                  </option>

                  <option value="4">
                    Particular Pagado
                  </option>

                  <option value="5">
                    Corporación de Administración Delegada
                  </option>

                  <option value="6">
                    Servicio Local de Educación Pública (SLEP)
                  </option>
                </select>
              </label>

              <label>
                Edad

                <input
                  type="number"
                  name="EDAD"
                  min="10"
                  max="130"
                  value={form.EDAD}
                  onChange={handleChange}
                  required
                />
              </label>

              <label>
                Género

                <select
                  name="GENERO"
                  value={form.GENERO}
                  onChange={handleChange}
                >
                  <option value="1">
                    Hombre
                  </option>

                  <option value="2">
                    Mujer
                  </option>
                </select>
              </label>

              <label>
                Nacionalidad

                <select
                  name="NACIONALIDAD"
                  value={form.NACIONALIDAD}
                  onChange={handleChange}
                >
                  <option value="0">Chileno</option>
                  <option value="1">Extranjero</option>
                  <option value="2">Nacionalizado</option>
                </select>
              </label>

            </div>

            <button disabled={loading}>
              {
                loading
                  ? "Analizando..."
                  : "Analizar perfil"
              }
            </button>

          </form>

          {
            error && (
              <div className="error">
                {error}
              </div>
            )
          }

        </div>

        {
          resultado && (
            <div className="card result-card">

              <span className="result-label">
                Probabilidad estimada
              </span>

              <div className="probability">
                {
                  resultado
                    .probabilidad_pct
                    .toFixed(2)
                }%
              </div>

              <div className="progress">
                <div
                  className="progress-fill"
                  style={{
                    width:
                      `${resultado.probabilidad_pct}%`
                  }}
                />
              </div>

              <p className="threshold">
                Umbral del modelo:{" "}
                {
                  (
                    resultado.threshold *
                    100
                  ).toFixed(0)
                }%
              </p>

              <div
                className={
                  resultado.sobre_umbral
                    ? "status positive"
                    : "status negative"
                }
              >
                {resultado.mensaje}
              </div>

              {
                explicacion && (
                  <div className="factors">

                    <h3>
                      ¿Qué factores influyeron?
                    </h3>

                    {
                      explicacion
                        .factores_principales
                        .map((factor) => (
                          <div
                            className="factor"
                            key={FEATURE_LABELS[factor.variable] || factor.variable}
                          >
                            <div>
                              <strong>
                                {FEATURE_LABELS[factor.variable] || factor.variable}
                              </strong>

                              <span>
                                Valor: {mostrarValor(
                                  factor.variable,
                                  factor.valor
                                )}
                              </span>

                              <small>
                                Intensidad:{" "}
                                <strong>
                                  {intensidadShap(
                                    factor.impacto_shap,
                                    explicacion.factores_principales
                                  )}
                                </strong>
                              </small>

                              <small>
                                Intensidad:{" "}
                                <strong>
                                  {intensidadShap(
                                    factor.impacto_shap,
                                    explicacion.factores_principales
                                  )}
                                </strong>
                              </small>
                            </div>

                            <span
                              className={
                                factor.direccion
                                === "aumenta"
                                  ? "impact up"
                                  : "impact down"
                              }
                            >
                              {
                                factor.direccion
                                === "aumenta"
                                  ? "↑ Aumenta la estimación"
                                  : "↓ Disminuye la estimación"
                              }
                            </span>
                          </div>
                        ))
                    }

                  </div>
                )
              }

              <div className="warning">
                {resultado.advertencia}
              </div>

              <div className="model-info">
                <h3>Acerca del modelo</h3>

                <div className="model-grid">
                  <div>
                    <strong>Modelo</strong>
                    <span>Random Forest</span>
                  </div>

                  <div>
                    <strong>Datos de desarrollo</strong>
                    <span>FUAS 2024</span>
                  </div>

                  <div>
                    <strong>Evaluación temporal</strong>
                    <span>FUAS 2025 · Holdout</span>
                  </div>

                  <div>
                    <strong>Umbral de clasificación</strong>
                    <span>42%</span>
                  </div>
                </div>

                <h4>Desempeño temporal 2025</h4>

                <div className="metrics-grid">
                  <div>
                    <strong>AUC</strong>
                    <span>0,788</span>
                  </div>

                  <div>
                    <strong>Balanced Accuracy</strong>
                    <span>0,710</span>
                  </div>

                  <div>
                    <strong>Recall</strong>
                    <span>0,851</span>
                  </div>

                  <div>
                    <strong>F1</strong>
                    <span>0,710</span>
                  </div>

                  <div>
                    <strong>FNR</strong>
                    <span>0,149</span>
                  </div>
                </div>

              </div>

              <div className="model-info">
                <h3>Acerca del modelo</h3>

                <div className="model-grid">
                  <div>
                    <strong>Modelo</strong>
                    <span>Random Forest</span>
                  </div>

                  <div>
                    <strong>Datos de desarrollo</strong>
                    <span>FUAS 2024</span>
                  </div>

                  <div>
                    <strong>Evaluación temporal</strong>
                    <span>FUAS 2025 · Holdout</span>
                  </div>

                  <div>
                    <strong>Umbral de clasificación</strong>
                    <span>42%</span>
                  </div>
                </div>

                <h4>Desempeño temporal 2025</h4>

                <div className="metrics-grid">
                  <div>
                    <strong>AUC</strong>
                    <span>0,788</span>
                  </div>

                  <div>
                    <strong>Balanced Accuracy</strong>
                    <span>0,710</span>
                  </div>

                  <div>
                    <strong>Recall</strong>
                    <span>0,851</span>
                  </div>

                  <div>
                    <strong>F1</strong>
                    <span>0,710</span>
                  </div>

                  <div>
                    <strong>FNR</strong>
                    <span>0,149</span>
                  </div>
                </div>
              </div>

            </div>
          )
        }

      </section>
      )}

      {vista === "dashboard" && (
        <section className="dashboard-view">

          <div className="dashboard-header">
            <div>
              <span className="dashboard-kicker">
                Datos históricos MINEDUC
              </span>

              <h2>
                Análisis histórico FUAS
              </h2>

              <p>
                Evolución descriptiva de postulaciones y
                asignaciones entre 2008 y 2025.
              </p>
            </div>

            <label className="year-selector">
              Año de análisis

              <select
                value={dashboardYear}
                onChange={(e) =>
                  setDashboardYear(
                    Number(e.target.value)
                  )
                }
              >
                {Array.from(
                  { length: 18 },
                  (_, i) => 2025 - i
                ).map((anio) => (
                  <option
                    key={anio}
                    value={anio}
                  >
                    {anio}
                  </option>
                ))}
              </select>
            </label>
          </div>

          {dashboardLoading && (
            <div className="dashboard-status">
              Cargando análisis histórico...
            </div>
          )}

          {dashboardError && (
            <div className="error">
              {dashboardError}
            </div>
          )}

          {dashboard && (
            <>
              {resumenSeleccionado && (
                <div className="dashboard-kpis">

                  <div className="kpi-card">
                    <span>Postulantes</span>
                    <strong>
                      {resumenSeleccionado
                        .postulantes_mrun
                        .toLocaleString("es-CL")}
                    </strong>
                  </div>

                  <div className="kpi-card">
                    <span>Beneficiarios</span>
                    <strong>
                      {resumenSeleccionado
                        .beneficio_valido
                        .toLocaleString("es-CL")}
                    </strong>
                  </div>

                  <div className="kpi-card">
                    <span>Tasa de beneficio</span>
                    <strong>
                      {resumenSeleccionado
                        .tasa_beneficio_pct
                        .toFixed(2)}%
                    </strong>
                  </div>

                </div>
              )}

              <div className="dashboard-card">
                <h3>
                  Evolución de la tasa de beneficio
                </h3>

                <p className="dashboard-caption">
                  Porcentaje de casos resueltos con al menos
                  un beneficio reconocido.
                </p>

                <div className="history-bars">
                  {resumenAnual.map((item) => (
                    <div
                      className="history-row"
                      key={item.anio}
                    >
                      <span className="history-year">
                        {item.anio}
                      </span>

                      <div className="history-track">
                        <div
                          className="history-fill"
                          style={{
                            width:
                              `${item.tasa_beneficio_pct}%`
                          }}
                        />
                      </div>

                      <strong>
                        {item.tasa_beneficio_pct
                          .toFixed(1)}%
                      </strong>
                    </div>
                  ))}
                </div>
              </div>

              <div className="dashboard-grid">

                <div className="dashboard-card">
                  <h3>
                    Tasa por género · {dashboardYear}
                  </h3>

                  {datosGenero.length > 0 ? (
                    datosGenero.map((item) => (
                      <div
                        className="category-row"
                        key={item.codigo}
                      >
                        <div>
                          <strong>
                            {item.categoria}
                          </strong>

                          <small>
                            n = {item.casos
                              .toLocaleString("es-CL")}
                          </small>
                        </div>

                        <span>
                          {item.tasa_beneficio_pct
                            .toFixed(1)}%
                        </span>
                      </div>
                    ))
                  ) : (
                    <p className="no-data">
                      Sin datos para este año.
                    </p>
                  )}
                </div>

                <div className="dashboard-card">
                  <h3>
                    Tasa por quintil · {dashboardYear}
                  </h3>

                  {datosQuintil.length > 0 ? (
                    datosQuintil.map((item) => (
                      <div
                        className="category-row"
                        key={item.quintil}
                      >
                        <div>
                          <strong>
                            Quintil {item.quintil}
                          </strong>

                          <small>
                            n = {item.casos
                              .toLocaleString("es-CL")}
                          </small>
                        </div>

                        <span>
                          {item.tasa_beneficio_pct
                            .toFixed(1)}%
                        </span>
                      </div>
                    ))
                  ) : (
                    <p className="no-data">
                      La variable QUINTIL_SE4 no está
                      disponible para este año.
                    </p>
                  )}
                </div>

              </div>

              <div className="dashboard-card">
                <h3>
                  Tasa por tramo de edad · {dashboardYear}
                </h3>

                <div className="age-grid">
                  {datosEdad.map((item) => (
                    <div
                      className="age-item"
                      key={item.tramo}
                    >
                      <span>{item.tramo}</span>

                      <strong>
                        {item.tasa_beneficio_pct
                          .toFixed(1)}%
                      </strong>

                      <small>
                        n = {item.casos
                          .toLocaleString("es-CL")}
                      </small>
                    </div>
                  ))}
                </div>
              </div>

              <div className="dashboard-methodology">
                <strong>
                  Consideraciones metodológicas
                </strong>

                <p>
                  {
                    dashboard.metadata
                      .nota_metodologica
                  }
                </p>

                <p>
                  {
                    dashboard.metadata
                      .definicion_beneficio
                  }
                </p>

                <p>
                  La disponibilidad de variables cambia
                  según el año. La ausencia de un gráfico
                  no debe interpretarse como valor cero.
                </p>
              </div>
            </>
          )}

        </section>
      )}

    </main>
  );
}

export default App;
