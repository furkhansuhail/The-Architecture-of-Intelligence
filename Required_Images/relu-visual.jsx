import { useState, useEffect } from "react";

const COLORS = {
  bg: "#0a0a0f",
  card: "#12121a",
  border: "#1e1e2e",
  accent: "#ff6b35",
  accentGlow: "rgba(255, 107, 53, 0.3)",
  blue: "#4ecdc4",
  blueGlow: "rgba(78, 205, 196, 0.3)",
  purple: "#a78bfa",
  purpleGlow: "rgba(167, 139, 250, 0.3)",
  yellow: "#fbbf24",
  yellowGlow: "rgba(251, 191, 36, 0.3)",
  text: "#e4e4e7",
  muted: "#71717a",
  dim: "#3f3f46",
};

// Neuron component
function Neuron({ x, y, value, active, label, color = COLORS.blue, delay = 0 }) {
  const [show, setShow] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setShow(true), delay);
    return () => clearTimeout(t);
  }, [delay]);

  const r = 22;
  const intensity = active ? 1 : 0.3;

  return (
    <g style={{ opacity: show ? 1 : 0, transition: "opacity 0.5s ease" }}>
      <circle cx={x} cy={y} r={r + 6} fill={active ? color.replace(")", ",0.15)").replace("rgb", "rgba") : "none"} />
      <circle cx={x} cy={y} r={r} fill={COLORS.card} stroke={color} strokeWidth={active ? 2.5 : 1} opacity={intensity} />
      {value !== undefined && (
        <text x={x} y={y + 1} textAnchor="middle" dominantBaseline="middle" fill={color} fontSize="11" fontWeight="600" fontFamily="'JetBrains Mono', monospace" opacity={intensity}>
          {typeof value === "number" ? value.toFixed(1) : value}
        </text>
      )}
      {label && (
        <text x={x} y={y + r + 16} textAnchor="middle" fill={COLORS.muted} fontSize="9" fontFamily="'JetBrains Mono', monospace">
          {label}
        </text>
      )}
    </g>
  );
}

// Connection line
function Connection({ x1, y1, x2, y2, weight, active, delay = 0 }) {
  const [show, setShow] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setShow(true), delay);
    return () => clearTimeout(t);
  }, [delay]);

  const opacity = active ? (weight > 0 ? 0.6 : 0.25) : 0.08;
  const color = weight > 0 ? COLORS.blue : COLORS.accent;

  return (
    <line
      x1={x1} y1={y1} x2={x2} y2={y2}
      stroke={color} strokeWidth={active ? Math.abs(weight) * 2 + 0.5 : 0.5}
      opacity={show ? opacity : 0}
      style={{ transition: "all 0.5s ease" }}
    />
  );
}

// ReLU activation visualization
function ReLUGraph({ x, y, width, height, inputVal, highlighted }) {
  const graphW = width;
  const graphH = height;
  const midX = x + graphW / 2;
  const midY = y + graphH / 2;
  const scale = graphH / 6;

  // Build path: flat at 0 for negatives, linear for positives
  const points = [];
  for (let i = -3; i <= 3; i += 0.1) {
    const px = midX + i * (graphW / 6);
    const py = midY - Math.max(0, i) * scale;
    points.push(`${px},${py}`);
  }

  const inputX = midX + inputVal * (graphW / 6);
  const outputVal = Math.max(0, inputVal);
  const inputY = midY - outputVal * scale;

  return (
    <g>
      {/* Background */}
      <rect x={x} y={y} width={graphW} height={graphH} rx="8" fill={highlighted ? "rgba(255,107,53,0.08)" : "rgba(255,255,255,0.02)"} stroke={highlighted ? COLORS.accent : COLORS.dim} strokeWidth={highlighted ? 2 : 1} />

      {/* Grid lines */}
      <line x1={x + 4} y1={midY} x2={x + graphW - 4} y2={midY} stroke={COLORS.dim} strokeWidth="0.5" />
      <line x1={midX} y1={y + 4} x2={midX} y2={y + graphH - 4} stroke={COLORS.dim} strokeWidth="0.5" />

      {/* ReLU curve */}
      <polyline points={points.join(" ")} fill="none" stroke={COLORS.accent} strokeWidth="2.5" strokeLinecap="round" />

      {/* Input dot */}
      {highlighted && (
        <>
          <line x1={inputX} y1={midY} x2={inputX} y2={inputY} stroke={COLORS.yellow} strokeWidth="1" strokeDasharray="3,3" opacity="0.6" />
          <circle cx={inputX} cy={inputY} r="5" fill={COLORS.yellow}>
            <animate attributeName="r" values="4;6;4" dur="1.5s" repeatCount="indefinite" />
          </circle>
        </>
      )}

      {/* Label */}
      <text x={x + graphW / 2} y={y - 8} textAnchor="middle" fill={highlighted ? COLORS.accent : COLORS.muted} fontSize="11" fontWeight="700" fontFamily="'JetBrains Mono', monospace">
        ReLU(x) = max(0, x)
      </text>
    </g>
  );
}

// Arrow showing data flow
function FlowArrow({ x1, y1, x2, y2, label, color = COLORS.muted, active = false }) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.sqrt(dx * dx + dy * dy);
  const nx = dx / len;
  const ny = dy / len;
  const arrowLen = 8;

  return (
    <g opacity={active ? 1 : 0.5}>
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={active ? 2 : 1.5} strokeDasharray={active ? "none" : "6,4"} />
      <polygon
        points={`${x2},${y2} ${x2 - nx * arrowLen - ny * 4},${y2 - ny * arrowLen + nx * 4} ${x2 - nx * arrowLen + ny * 4},${y2 - ny * arrowLen - nx * 4}`}
        fill={color}
      />
      {label && (
        <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 10} textAnchor="middle" fill={color} fontSize="10" fontWeight="600" fontFamily="'JetBrains Mono', monospace">
          {label}
        </text>
      )}
    </g>
  );
}

// Stage label box
function StageBox({ x, y, width, height, title, subtitle, color, active }) {
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} rx="6" fill={active ? `${color}15` : "rgba(255,255,255,0.02)"} stroke={active ? color : COLORS.dim} strokeWidth={active ? 2 : 1} />
      <text x={x + width / 2} y={y + height / 2 - (subtitle ? 6 : 0)} textAnchor="middle" dominantBaseline="middle" fill={active ? color : COLORS.muted} fontSize="12" fontWeight="700" fontFamily="'JetBrains Mono', monospace">
        {title}
      </text>
      {subtitle && (
        <text x={x + width / 2} y={y + height / 2 + 10} textAnchor="middle" dominantBaseline="middle" fill={COLORS.muted} fontSize="9" fontFamily="'JetBrains Mono', monospace">
          {subtitle}
        </text>
      )}
    </g>
  );
}

export default function ReLUVisual() {
  const [step, setStep] = useState(0);
  const [autoPlay, setAutoPlay] = useState(false);
  const [inputVal, setInputVal] = useState(1.5);

  const steps = [
    { id: 0, title: "Input Layer", desc: "Raw data enters the network. Each neuron holds one feature value (e.g., pixel intensity, sensor reading)." },
    { id: 1, title: "Linear Transform", desc: "Weights multiply inputs, biases are added: z = Wx + b. This is just matrix multiplication — purely linear." },
    { id: 2, title: "⚡ ReLU Activation", desc: "ReLU(z) = max(0, z). This is where the magic happens — it introduces non-linearity, killing negative values and passing positives unchanged." },
    { id: 3, title: "Next Layer", desc: "The activated outputs become inputs to the next layer. Without ReLU, stacking layers would collapse into a single linear transformation." },
    { id: 4, title: "Full Picture", desc: "Every hidden layer repeats: Linear → ReLU → Linear → ReLU → ... → Output. ReLU sits BETWEEN every pair of linear transformations." },
  ];

  useEffect(() => {
    if (!autoPlay) return;
    const t = setInterval(() => setStep((s) => (s + 1) % steps.length), 3000);
    return () => clearInterval(t);
  }, [autoPlay]);

  // Example values flowing through
  const inputs = [0.8, -0.3, 1.5, -0.7];
  const weights = [[0.5, -0.2, 0.8, 0.1], [0.3, 0.7, -0.4, 0.6], [-0.6, 0.4, 0.9, -0.3]];
  const linearOutputs = weights.map((w) => w.reduce((s, wi, i) => s + wi * inputs[i], 0));
  const reluOutputs = linearOutputs.map((v) => Math.max(0, v));

  return (
    <div style={{ background: COLORS.bg, minHeight: "100vh", padding: "24px", fontFamily: "'JetBrains Mono', 'SF Mono', monospace", color: COLORS.text }}>
      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: "28px" }}>
        <h1 style={{ fontSize: "28px", fontWeight: "800", margin: 0, background: `linear-gradient(135deg, ${COLORS.accent}, ${COLORS.yellow})`, WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          Where Does ReLU Fit?
        </h1>
        <p style={{ color: COLORS.muted, fontSize: "13px", marginTop: "6px" }}>
          Inside every hidden layer of a neural network
        </p>
      </div>

      {/* Main SVG Diagram */}
      <div style={{ display: "flex", justifyContent: "center", marginBottom: "20px" }}>
        <svg viewBox="0 0 900 420" style={{ width: "100%", maxWidth: "900px", background: "rgba(255,255,255,0.01)", borderRadius: "12px", border: `1px solid ${COLORS.border}` }}>

          {/* Stage labels at top */}
          <StageBox x={30} y={15} width={120} height={38} title="INPUT" subtitle="features" color={COLORS.blue} active={step === 0 || step === 4} />
          <StageBox x={210} y={15} width={140} height={38} title="LINEAR" subtitle="z = Wx + b" color={COLORS.purple} active={step === 1 || step === 4} />
          <StageBox x={420} y={15} width={140} height={38} title="ReLU ⚡" subtitle="max(0, z)" color={COLORS.accent} active={step === 2 || step === 4} />
          <StageBox x={630} y={15} width={140} height={38} title="OUTPUT" subtitle="to next layer" color={COLORS.yellow} active={step === 3 || step === 4} />

          {/* Flow arrows between stages */}
          <FlowArrow x1={155} y1={34} x2={205} y2={34} color={COLORS.dim} active={step >= 1} />
          <FlowArrow x1={355} y1={34} x2={415} y2={34} color={COLORS.dim} active={step >= 2} />
          <FlowArrow x1={565} y1={34} x2={625} y2={34} color={COLORS.dim} active={step >= 3} />

          {/* INPUT NEURONS */}
          {inputs.map((v, i) => {
            const ny = 110 + i * 70;
            return <Neuron key={`in-${i}`} x={90} y={ny} value={v} active={step >= 0} label={`x${i + 1}`} color={COLORS.blue} delay={i * 80} />;
          })}

          {/* Connections: Input → Linear */}
          {step >= 1 && inputs.map((_, i) =>
            linearOutputs.map((_, j) => (
              <Connection key={`c1-${i}-${j}`} x1={112} y1={110 + i * 70} x2={258} y2={130 + j * 85} weight={weights[j][i]} active={step >= 1} delay={i * 30 + j * 30} />
            ))
          )}

          {/* LINEAR OUTPUT NEURONS (pre-activation) */}
          {linearOutputs.map((v, i) => {
            const ny = 130 + i * 85;
            return <Neuron key={`lin-${i}`} x={280} y={ny} value={v} active={step >= 1} label={`z${i + 1}`} color={COLORS.purple} delay={200 + i * 100} />;
          })}

          {/* ReLU Graph */}
          <ReLUGraph x={390} y={90} width={170} height={120} inputVal={linearOutputs[0]} highlighted={step === 2 || step === 4} />

          {/* Mini ReLU indicators for each neuron */}
          {linearOutputs.map((v, i) => {
            const ny = 130 + i * 85;
            const out = Math.max(0, v);
            const passed = v > 0;
            return step >= 2 ? (
              <g key={`relu-${i}`}>
                {/* Arrow from linear to relu result */}
                <line x1={302} y1={ny} x2={570} y2={ny} stroke={passed ? COLORS.accent : COLORS.dim} strokeWidth={passed ? 1.5 : 0.8} strokeDasharray={passed ? "none" : "4,4"} opacity={passed ? 0.7 : 0.3} />
                {!passed && (
                  <g>
                    <line x1={428} y1={ny - 8} x2={442} y2={ny + 8} stroke="#ef4444" strokeWidth="2.5" opacity="0.8" />
                    <line x1={442} y1={ny - 8} x2={428} y2={ny + 8} stroke="#ef4444" strokeWidth="2.5" opacity="0.8" />
                  </g>
                )}
                {passed && (
                  <text x={435} y={ny - 6} textAnchor="middle" fill={COLORS.accent} fontSize="9" fontWeight="700" opacity="0.9">✓</text>
                )}
              </g>
            ) : null;
          })}

          {/* RELU OUTPUT NEURONS */}
          {reluOutputs.map((v, i) => {
            const ny = 130 + i * 85;
            return <Neuron key={`relu-out-${i}`} x={590} y={ny} value={v} active={step >= 2} label={`a${i + 1}`} color={v > 0 ? COLORS.accent : COLORS.dim} delay={400 + i * 100} />;
          })}

          {/* Connections to next layer */}
          {step >= 3 && reluOutputs.map((v, i) => {
            const ny = 130 + i * 85;
            return [0, 1].map((j) => (
              <Connection key={`c2-${i}-${j}`} x1={612} y1={ny} x2={728} y2={150 + j * 100} weight={v > 0 ? 0.5 : 0.1} active={step >= 3 && v > 0} delay={i * 50} />
            ));
          })}

          {/* NEXT LAYER NEURONS */}
          {[0, 1].map((i) => (
            <Neuron key={`next-${i}`} x={750} y={150 + i * 100} value="?" active={step >= 3} label={`h${i + 1}`} color={COLORS.yellow} delay={600 + i * 100} />
          ))}

          {/* Big bracket showing "ONE HIDDEN LAYER" */}
          {step === 4 && (
            <g>
              <rect x={195} y={68} width={430} height={310} rx="10" fill="none" stroke={COLORS.accent} strokeWidth="1.5" strokeDasharray="8,6" opacity="0.4" />
              <text x={410} y={395} textAnchor="middle" fill={COLORS.accent} fontSize="12" fontWeight="700" opacity="0.7">
                ← One Hidden Layer = Linear + ReLU →
              </text>
            </g>
          )}

          {/* Data flow annotation */}
          <text x={450} y={408} textAnchor="middle" fill={COLORS.dim} fontSize="9">
            {step < 4 ? "Click steps below to trace the data flow →" : "This pattern repeats for every hidden layer in the network"}
          </text>
        </svg>
      </div>

      {/* Step controls */}
      <div style={{ display: "flex", justifyContent: "center", gap: "8px", marginBottom: "20px", flexWrap: "wrap" }}>
        {steps.map((s) => (
          <button
            key={s.id}
            onClick={() => { setStep(s.id); setAutoPlay(false); }}
            style={{
              padding: "8px 16px",
              borderRadius: "8px",
              border: `1.5px solid ${step === s.id ? COLORS.accent : COLORS.border}`,
              background: step === s.id ? `${COLORS.accent}20` : COLORS.card,
              color: step === s.id ? COLORS.accent : COLORS.muted,
              cursor: "pointer",
              fontSize: "12px",
              fontWeight: "600",
              fontFamily: "inherit",
              transition: "all 0.2s",
            }}
          >
            {s.id + 1}. {s.title}
          </button>
        ))}
        <button
          onClick={() => setAutoPlay(!autoPlay)}
          style={{
            padding: "8px 14px",
            borderRadius: "8px",
            border: `1.5px solid ${autoPlay ? COLORS.yellow : COLORS.border}`,
            background: autoPlay ? `${COLORS.yellow}20` : COLORS.card,
            color: autoPlay ? COLORS.yellow : COLORS.muted,
            cursor: "pointer",
            fontSize: "12px",
            fontFamily: "inherit",
          }}
        >
          {autoPlay ? "⏸ Pause" : "▶ Auto"}
        </button>
      </div>

      {/* Explanation card */}
      <div style={{ maxWidth: "700px", margin: "0 auto 24px", padding: "18px 22px", background: COLORS.card, borderRadius: "10px", border: `1px solid ${step === 2 ? COLORS.accent : COLORS.border}`, transition: "border 0.3s" }}>
        <div style={{ fontSize: "15px", fontWeight: "700", color: step === 2 ? COLORS.accent : COLORS.text, marginBottom: "6px" }}>
          Step {step + 1}: {steps[step].title}
        </div>
        <div style={{ fontSize: "13px", color: COLORS.muted, lineHeight: "1.6" }}>
          {steps[step].desc}
        </div>
      </div>

      {/* Interactive ReLU explorer */}
      <div style={{ maxWidth: "700px", margin: "0 auto", padding: "20px 22px", background: COLORS.card, borderRadius: "10px", border: `1px solid ${COLORS.border}` }}>
        <div style={{ fontSize: "13px", fontWeight: "700", color: COLORS.accent, marginBottom: "12px" }}>
          🧪 Try it: Slide to see ReLU in action
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <span style={{ fontSize: "11px", color: COLORS.muted, minWidth: "30px" }}>-3.0</span>
          <input
            type="range" min={-3} max={3} step={0.1} value={inputVal}
            onChange={(e) => setInputVal(parseFloat(e.target.value))}
            style={{ flex: 1, accentColor: COLORS.accent }}
          />
          <span style={{ fontSize: "11px", color: COLORS.muted, minWidth: "30px" }}>3.0</span>
        </div>
        <div style={{ display: "flex", justifyContent: "center", gap: "40px", marginTop: "14px" }}>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: "10px", color: COLORS.muted, marginBottom: "4px" }}>INPUT (z)</div>
            <div style={{ fontSize: "22px", fontWeight: "800", color: inputVal < 0 ? "#ef4444" : COLORS.blue }}>
              {inputVal.toFixed(1)}
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", fontSize: "20px", color: COLORS.accent }}>→</div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: "10px", color: COLORS.muted, marginBottom: "4px" }}>ReLU</div>
            <div style={{ fontSize: "14px", color: COLORS.dim }}>max(0, {inputVal.toFixed(1)})</div>
          </div>
          <div style={{ display: "flex", alignItems: "center", fontSize: "20px", color: COLORS.accent }}>→</div>
          <div style={{ textAlign: "center" }}>
            <div style={{ fontSize: "10px", color: COLORS.muted, marginBottom: "4px" }}>OUTPUT (a)</div>
            <div style={{ fontSize: "22px", fontWeight: "800", color: Math.max(0, inputVal) === 0 ? "#ef4444" : COLORS.accent }}>
              {Math.max(0, inputVal).toFixed(1)}
            </div>
          </div>
        </div>
        {inputVal < 0 && (
          <div style={{ textAlign: "center", marginTop: "10px", fontSize: "11px", color: "#ef4444", opacity: 0.8 }}>
            ☠ Negative value killed! This neuron is "dead" for this input.
          </div>
        )}
        {inputVal > 0 && (
          <div style={{ textAlign: "center", marginTop: "10px", fontSize: "11px", color: COLORS.accent, opacity: 0.8 }}>
            ✓ Positive value passes through unchanged!
          </div>
        )}
      </div>

      {/* Key Insight */}
      <div style={{ maxWidth: "700px", margin: "20px auto 0", padding: "16px 22px", background: "rgba(255,107,53,0.06)", borderRadius: "10px", border: `1px solid rgba(255,107,53,0.2)` }}>
        <div style={{ fontSize: "12px", fontWeight: "700", color: COLORS.accent, marginBottom: "6px" }}>💡 Key Insight</div>
        <div style={{ fontSize: "12px", color: COLORS.muted, lineHeight: "1.7" }}>
          ReLU sits <span style={{ color: COLORS.accent, fontWeight: "700" }}>after every linear transformation</span> and <span style={{ color: COLORS.accent, fontWeight: "700" }}>before the next layer's input</span>. Without it, a 100-layer network would mathematically reduce to a single linear equation. ReLU is what gives deep networks the ability to learn complex, non-linear patterns.
        </div>
      </div>
    </div>
  );
}
