import React, { memo, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { AnimatePresence, motion } from "framer-motion";
import {
  BadgeCheck, BrainCircuit, CheckCircle2, ChevronDown, ChevronRight, CircleAlert,
  Copy, Database, Eye, EyeOff, FileCheck2, FileText, Gauge, KeyRound, Layers3,
  LoaderCircle, LockKeyhole, Moon, PanelLeftClose, PanelLeftOpen, Plus, Search,
  Send, ShieldCheck, Sparkles, Sun, Trash2, Upload, UserRound, WandSparkles, X,
  LogIn, LogOut, Mail, UserPlus, ArrowRight, FileAudio, FileVideo, Image as ImageIcon,
  History, Pencil, MoreHorizontal, Home, BookOpen, Settings2, Paperclip, Grid2X2,
  Download, ExternalLink, ThumbsUp, ThumbsDown, Link2, SlidersHorizontal, RefreshCcw,
  FileSpreadsheet, PanelRightClose, PanelRightOpen
} from "lucide-react";
import "./styles.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const providers = {
  deepseek: { name: "DeepSeek", short: "DS", note: "Fast reasoning via DeepSeek API", model: "deepseek-chat", base: "https://api.deepseek.com" },
  gemini: { name: "Google Gemini", short: "G", note: "Fast multimodal models via Gemini API", model: "gemini-3.6-flash", base: "https://generativelanguage.googleapis.com/v1beta/openai" },
  openai: { name: "OpenAI", short: "AI", note: "GPT models via your key", model: "gpt-4.1-mini", base: "https://api.openai.com/v1" },
  anthropic: { name: "Anthropic", short: "A", note: "Claude models via your key", model: "claude-haiku-4-5", base: "" },
  groq: { name: "Groq", short: "GQ", note: "Ultra-fast OpenAI-compatible API", model: "llama-3.3-70b-versatile", base: "https://api.groq.com/openai/v1" },
  openrouter: { name: "OpenRouter", short: "OR", note: "One key, many model families", model: "openai/gpt-4.1-mini", base: "https://openrouter.ai/api/v1" },
  ollama: { name: "Ollama", short: "OL", note: "Run a model fully on your machine", model: "llama3.2", base: "http://localhost:11434/v1" },
  custom: { name: "Custom", short: "<>" , note: "Any OpenAI-compatible endpoint", model: "", base: "" },
};

const keyStoreName = (provider) => `axiom-key-${provider}`;
const readSessionKey = (provider) => {
  try { return sessionStorage.getItem(keyStoreName(provider)) || ""; } catch { return ""; }
};
const saveSessionKey = (provider, key) => {
  try { key ? sessionStorage.setItem(keyStoreName(provider), key) : sessionStorage.removeItem(keyStoreName(provider)); } catch {}
};

function SecureKeyField({ provider, value, onChange }) {
  const [draft, setDraft] = useState("");
  const hasKey = Boolean(value);

  const commitValue = (raw) => {
    const next = (raw || "").trim();
    if (!next) return;
    onChange(next);
    saveSessionKey(provider, next);
    setDraft("");
  };

  const clear = () => {
    onChange("");
    saveSessionKey(provider, "");
    setDraft("");
  };

  return (
    <div className={`secureKey ${hasKey ? "saved" : ""}`}>
      <div className="secureKeyInput">
        <KeyRound size={15}/>
        <input
          type="password"
          autoComplete="new-password"
          value={draft}
          onChange={e=>setDraft(e.target.value)}
          onPaste={e=>{
            const pasted = e.clipboardData?.getData("text") || "";
            if (pasted.trim()) {
              e.preventDefault();
              commitValue(pasted);
            }
          }}
          onKeyDown={e=>{
            if(e.key === "Enter"){
              e.preventDefault();
              commitValue(draft);
            }
          }}
          placeholder={hasKey ? "API key saved" : "Paste API key"}
          aria-label="Provider API key"
        />
        {!hasKey && draft ? (
          <button className="keySave" type="button" onClick={()=>commitValue(draft)}>Save</button>
        ) : hasKey ? (
          <span className="keySaved"><CheckCircle2 size={13}/> Saved</span>
        ) : (
          <span className="keyPrivate"><LockKeyhole size={12}/></span>
        )}
      </div>
      {hasKey && <button className="keyClear" type="button" onClick={clear}>Clear</button>}
    </div>
  );
}


const StarBackground = memo(function StarBackground({ busy }) {
  const makeStars = (count = 74) => Array.from({ length: count }, (_, i) => ({
    id: `${Date.now()}-${i}-${Math.random()}`,
    left: Math.random() * 100,
    top: Math.random() * 100,
    size: 0.7 + Math.random() * 1.9,
    opacity: 0.18 + Math.random() * 0.72,
    delay: Math.random() * 4.5,
    duration: 1.9 + Math.random() * 4.8,
  }));

  const makePulse = () => ({
    id: `${Date.now()}-${Math.random()}`,
    left: 8 + Math.random() * 84,
    top: 8 + Math.random() * 78,
    size: 18 + Math.random() * 32,
  });

  const [stars, setStars] = useState(() => makeStars());
  const [pulse, setPulse] = useState(() => makePulse());

  useEffect(() => {
    const starTimer = setInterval(() => {
      setStars(makeStars());
    }, busy ? 3600 : 5600);
    return () => clearInterval(starTimer);
  }, [busy]);

  useEffect(() => {
    const pulseTimer = setInterval(() => {
      setPulse(makePulse());
    }, busy ? 2100 : 4800);
    return () => clearInterval(pulseTimer);
  }, [busy]);

  return (
    <div className={`spaceBackground ${busy ? "isBusy" : ""}`} aria-hidden="true">
      <div className="starField">
        {stars.map(star => (
          <span
            key={star.id}
            className="randomStar"
            style={{
              left: `${star.left}%`,
              top: `${star.top}%`,
              width: `${star.size}px`,
              height: `${star.size}px`,
              opacity: star.opacity,
              animationDelay: `${star.delay}s`,
              animationDuration: `${star.duration}s`,
            }}
          />
        ))}
      </div>
      <span
        key={pulse.id}
        className="signalPulse"
        style={{ left: `${pulse.left}%`, top: `${pulse.top}%`, width: pulse.size, height: pulse.size }}
      />
      <span className="driftSpark driftSparkA" />
      <span className="driftSpark driftSparkB" />
      <span className="driftSpark driftSparkC" />
    </div>
  );
});

function LogoMark({ hero = false, active = false, compact = false }) {
  return (
    <div className={`logoMark axiomOrb ${hero ? "heroLogo" : ""} ${active ? "isThinking" : ""} ${compact ? "compact" : ""}`} role="img" aria-label="Axiom">
      <span className="axiomHalo" />
      <span className="axiomGlyph" aria-hidden="true" />
    </div>
  );
}

function Brand() {
  return (
    <div className="brand">
      <LogoMark />
      <div>
        <div className="brandTitle">TrustRAG</div>
        <div className="brandSub">RAG-1 Axiom · evidence-aware intelligence</div>
      </div>
    </div>
  );
}

function ProviderBadge({ provider }) {
  const p = providers[provider] || providers.custom;
  return <span className={`providerBadge provider-${provider}`}>{p.short}</span>;
}

function ThemeToggle({ theme, setTheme }) {
  return (
    <div className="themeToggle" role="group" aria-label="Theme">
      <button className={theme === "dark" ? "active" : ""} onClick={() => setTheme("dark")}><Moon size={14}/> Dark</button>
      <button className={theme === "light" ? "active" : ""} onClick={() => setTheme("light")}><Sun size={14}/> Light</button>
    </div>
  );
}

function Sidebar({
  open, onClose, llm, setLlm, topK, setTopK, sourceMode, setSourceMode,
  rowIndex, setRowIndex, loadDatasetRow, datasetInfo, documentText,
  setDocumentText, setSuggestedQuestion, uploadedFile, theme, setTheme,
  user, chats, currentChatId, onOpenChat, onDeleteChat, onRenameChat,
  historySearch, setHistorySearch, onUploadFile, uploadError, onClearFile
}) {
  const fileRef = useRef(null);
  const p = providers[llm.provider];

  const changeProvider = (provider) => {
    const next = providers[provider];
    setLlm({ provider, api_key: readSessionKey(provider), model: next.model, base_url: next.base });
  };

  const chooseFile = (file) => {
    if (!file) return;
    onUploadFile(file);
    if (fileRef.current) fileRef.current.value = "";
  };

  const visibleChats = chats.filter(chat =>
    chat.title.toLowerCase().includes(historySearch.toLowerCase())
  );

  return (
    <AnimatePresence>
      {open && (
        <motion.aside className="sidebar" initial={{ x: -28, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: -28, opacity: 0 }} transition={{ duration: .2 }}>
          <div className="sideTop">
            <Brand />
            <button className="iconBtn mobileClose" onClick={onClose}><X size={17}/></button>
          </div>

          <div className="sideScroll">
            {user && <section className="sideSection historySection">
              <div className="sectionLabel"><History size={14}/> Chat history</div>
              <div className="historySearch"><Search size={13}/><input value={historySearch} onChange={e=>setHistorySearch(e.target.value)} placeholder="Search chats"/></div>
              <div className="historyList">
                {visibleChats.length === 0 ? <div className="historyEmpty">No saved chats yet.</div> : visibleChats.map(chat => (
                  <div className={`historyItem ${currentChatId === chat.id ? "active" : ""}`} key={chat.id}>
                    <button className="historyOpen" onClick={()=>onOpenChat(chat.id)} title={chat.title}>
                      <span>{chat.title}</span><small>{chat.source_name || (chat.source_mode === "dataset" ? "Dataset" : "Text source")}</small>
                    </button>
                    <div className="historyActions">
                      <button onClick={()=>onRenameChat(chat)} title="Rename"><Pencil size={12}/></button>
                      <button onClick={()=>onDeleteChat(chat)} title="Delete"><Trash2 size={12}/></button>
                    </div>
                  </div>
                ))}
              </div>
            </section>}

            <section className="sideSection">
              <div className="sectionLabel"><BrainCircuit size={14}/> AI connection</div>
              <label>Provider</label>
              <div className="providerSelectRow">
                <ProviderBadge provider={llm.provider}/>
                <div className="selectWrap">
                  <select value={llm.provider} onChange={e => changeProvider(e.target.value)}>
                    {Object.entries(providers).map(([id, item]) => <option key={id} value={id}>{item.name}</option>)}
                  </select>
                  <ChevronDown size={15}/>
                </div>
              </div>
              <div className="providerHint"><span className="onlineDot"/>{p.note}</div>

              {llm.provider !== "ollama" && <>
                <label>API key</label>
                <SecureKeyField provider={llm.provider} value={llm.api_key} onChange={api_key=>setLlm({...llm, api_key})}/>
                <div className="privacyNote"><LockKeyhole size={12}/> Hidden after entry · kept only in this browser tab.</div>
              </>}

              <label>Model</label>
              <input className="field" value={llm.model} onChange={e=>setLlm({...llm, model:e.target.value})} placeholder="Model name"/>
              {(llm.provider === "custom" || llm.provider === "ollama") && <>
                <label>Base URL</label>
                <input className="field" value={llm.base_url} onChange={e=>setLlm({...llm, base_url:e.target.value})} placeholder="https://.../v1"/>
              </>}
            </section>

            <section className="sideSection">
              <div className="sectionLabel"><Database size={14}/> Knowledge source</div>
              <div className="sourceTabs">
                <button className={sourceMode==="dataset"?"active":""} onClick={()=>setSourceMode("dataset")}><Database size={14}/> Dataset</button>
                <button className={sourceMode==="paste"?"active":""} onClick={()=>setSourceMode("paste")}><FileText size={14}/> Text</button>
                <button className={sourceMode==="file"?"active":""} onClick={()=>setSourceMode("file")}><Upload size={14}/> File</button>
              </div>

              {sourceMode === "dataset" && <div className="rowControl">
                <div><label>FACTS row</label><div className="muted">0–{datasetInfo?.max_index ?? 859}</div></div>
                <input type="number" min="0" max={datasetInfo?.max_index ?? 859} value={rowIndex} onChange={e=>setRowIndex(Number(e.target.value))}/>
                <button onClick={loadDatasetRow}>Load</button>
              </div>}

              {sourceMode === "paste" && <textarea className="sourceInput" value={documentText} onChange={e=>setDocumentText(e.target.value)} placeholder="Paste the source TrustRAG is allowed to use…"/>}

              {sourceMode === "file" && <>
                <input ref={fileRef} type="file" accept=".txt,.md,.csv,.json,.pdf,image/*,audio/*,video/*" hidden onChange={e=>chooseFile(e.target.files?.[0])}/>
                {!uploadedFile ? (
                  <button className="uploadBox" onClick={()=>fileRef.current?.click()}>
                    <span className="uploadIcon"><Upload size={19}/></span>
                    <span>Upload evidence</span>
                    <small>Text · PDF · Image · Audio · Video</small>
                  </button>
                ) : (
                  <div className="uploadedFileCard">
                    <div className="fileVisual"><FileCheck2 size={20}/></div>
                    <div className="fileMeta"><strong title={uploadedFile.name}>{uploadedFile.name}</strong><span>{uploadedFile.status === "processing" ? "Extracting evidence…" : `${(uploadedFile.size / 1024).toFixed(1)} KB · ${uploadedFile.characters?.toLocaleString() || ""} chars loaded`}</span></div>
                    <button type="button" className="fileRemove" onClick={onClearFile} title="Remove file"><Trash2 size={15}/></button>
                  </div>
                )}
              </>}
              {uploadError && <div className="inlineError"><CircleAlert size={13}/><span>{uploadError}</span></div>}
            </section>

            <section className="sideSection compact">
              <div className="topKHead">
                <div><div className="sectionLabel"><Layers3 size={14}/> Retrieval depth</div><div className="muted">Number of source chunks</div></div>
                <strong>{topK}</strong>
              </div>
              <input className="range" type="range" min="1" max="8" value={topK} onChange={e=>setTopK(Number(e.target.value))}/>
              <div className="rangeLabels"><span>Focused</span><span>Broad</span></div>
            </section>
          </div>

          <div className="sideBottom">
            <ThemeToggle theme={theme} setTheme={setTheme}/>
            <div className="sideFooter"><ShieldCheck size={14}/><span>GroundCheck audits evidence-backed answers</span></div>
          </div>
        </motion.aside>
      )}
    </AnimatePresence>
  );
}

function ScoreRing({ value = 0 }) {
  const pct = Math.round(value * 100);
  return <div className="scoreRing" style={{"--pct":`${pct*3.6}deg`}}><div><strong>{pct}%</strong><span>faithful</span></div></div>;
}

function VerificationPanel({ result, close }) {
  if (!result) return null;
  const gc = result.groundcheck;
  if (!gc) return null;
  const supported = gc.claims.filter(c => c.verdict === "SUPPORTED").length;
  const unsupported = gc.claims.filter(c => c.verdict === "UNSUPPORTED").length;
  const contradictions = gc.claims.filter(c => c.verdict === "CONTRADICTION").length;

  return (
    <motion.aside className="verifyPanel" initial={{ x: 28, opacity: 0 }} animate={{ x: 0, opacity: 1 }} exit={{ x: 28, opacity: 0 }} transition={{duration:.2}}>
      <div className="verifyHead">
        <div><span className="eyebrow"><ShieldCheck size={13}/> GroundCheck</span><h3>Verification report</h3></div>
        <button className="iconBtn" onClick={close}><X size={18}/></button>
      </div>
      <div className="verdictCard"><ScoreRing value={gc.faithfulness_score}/><div><div className={`verdict ${gc.overall_verdict.toLowerCase()}`}>{gc.overall_verdict.replaceAll("_"," ")}</div><p>Audited against retrieved evidence, not outside knowledge.</p></div></div>
      <div className="verificationNumbers">
        <div><span>Supported</span><strong>{supported}</strong><i className="good"/></div>
        <div><span>Unsupported</span><strong>{unsupported}</strong><i className="warn"/></div>
        <div><span>Contradictions</span><strong>{contradictions}</strong><i className="bad"/></div>
      </div>
      <div className="miniStats">
        <div><span>Generation</span><strong>{result.generation_time}s</strong></div>
        <div><span>Provider</span><strong>{providers[result.backend]?.name || result.backend}</strong></div>
        <div><span>Evidence</span><strong>{result.retrieved.length} chunks</strong></div>
      </div>
      <div className="claimList">{gc.claims.map((c,i)=><div className={`claim ${c.verdict.toLowerCase()}`} key={i}><div className="claimTop"><span>{c.verdict === "SUPPORTED" ? <CheckCircle2 size={15}/> : <CircleAlert size={15}/>} Claim {i+1}</span><b>{Math.round(c.nli.entailment*100)}% entail.</b></div><p>{c.claim}</p>{c.explanation && <small>{c.explanation}</small>}</div>)}</div>
      <details className="evidence"><summary>Retrieved evidence <ChevronRight size={14}/></summary>{result.retrieved.map((r,i)=><div key={i}><b>#{i+1} · {r.score.toFixed(3)}</b><p>{r.text}</p></div>)}</details>
    </motion.aside>
  );
}

const Message = memo(function Message({ msg, onInspect }) {
  const isAI = msg.role === "assistant";
  return (
    <motion.div className={`message ${isAI?"assistant":"user"}`} initial={{ opacity: 0, y: 9 }} animate={{ opacity: 1, y: 0 }} transition={{duration:.16}}>
      <div className="avatar">{isAI?<LogoMark compact/>:<UserRound size={17}/>}</div>
      <div className="bubble">
        <div className="messageMeta"><strong>{isAI?"RAG-1 Axiom":"You"}</strong>{isAI && msg.result && <span className={`modeBadge mode-${msg.result.mode || "general"}`}>{(msg.result.mode || "general").toUpperCase()}</span>}{isAI && msg.result?.groundcheck && <button onClick={()=>onInspect(msg.result)} title="Evidence faithfulness score"><ShieldCheck size={14}/>{Math.round(msg.result.groundcheck.faithfulness_score*100)}% faithful</button>}</div>
        {!isAI && msg.attachment && <div className="chatAttachment"><AttachmentVisual attachment={msg.attachment} size="large"/><div><strong>{msg.attachment.name}</strong><span>{msg.attachment.characters ? `${msg.attachment.characters.toLocaleString()} characters extracted` : "Knowledge source attached"}</span></div><CheckCircle2 size={15}/></div>}
        <div className={`messageText ${isAI ? "markdown-body" : ""}`}>
          {isAI ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {msg.text}
            </ReactMarkdown>
          ) : (
            msg.text
          )}
        </div>
        {isAI && <div className="messageActions"><button onClick={()=>navigator.clipboard.writeText(msg.text)}><Copy size={14}/> Copy</button>{msg.result?.groundcheck && <button onClick={()=>onInspect(msg.result)}><Gauge size={14}/> Verification</button>}</div>}
      </div>
    </motion.div>
  );
});

function WorkingTimeline() {
  const labels = ["Retrieving evidence", "Reasoning across context", "GroundCheck verification", "Composing response"];
  const [active, setActive] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => setActive(v => Math.min(v + 1, labels.length - 1)), 1050);
    return () => clearInterval(timer);
  }, []);

  return (
    <motion.div className="axiomThinking" initial={{opacity:0,y:6}} animate={{opacity:1,y:0}}>
      <LogoMark active compact/>
      <div className="thinkingCopy">
        <strong>Axiom is thinking<span className="thinkingDots"><i/><i/><i/></span></strong>
        <span>{labels[active]}</span>
      </div>
    </motion.div>
  );
}

function attachmentKind(type = "") {
  if (type.startsWith("image/")) return "image";
  if (type.startsWith("audio/")) return "audio";
  if (type.startsWith("video/")) return "video";
  return "file";
}

function AttachmentVisual({ attachment, size = "normal" }) {
  const kind = attachment?.kind || attachmentKind(attachment?.type || "");
  if (kind === "image" && attachment?.preview_url) {
    return <img className={`attachmentPreview ${size}`} src={attachment.preview_url} alt="" />;
  }
  const Icon = kind === "image" ? ImageIcon : kind === "audio" ? FileAudio : kind === "video" ? FileVideo : FileText;
  return <div className={`attachmentIcon ${size}`}><Icon size={size === "large" ? 20 : 17}/></div>;
}

function documentLabel(sourceMode,rowIndex,uploadedFile){ return sourceMode==="dataset"?`FACTS #${rowIndex}`:sourceMode==="file"?(uploadedFile?.name || "No file"):"Pasted text"; }

function Composer({ busy, sourceMode, rowIndex, llmProvider, topK, axiomMode, suggestedQuestion, uploadedFile, attachmentPending, onAsk, onUpload, onClearUpload }) {
  const [value, setValue] = useState("");
  useEffect(() => { if (suggestedQuestion) setValue(suggestedQuestion); }, [suggestedQuestion]);

  const submit = () => {
    const q = value.trim();
    if (!q || busy) return;
    setValue("");
    onAsk(q);
  };

  return (
    <div className="composerShell">
      {uploadedFile && attachmentPending && <motion.div className="composerAttachment" initial={{opacity:0,y:6,scale:.985}} animate={{opacity:1,y:0,scale:1}}><AttachmentVisual attachment={uploadedFile}/><div><strong title={uploadedFile.name}>{uploadedFile.name}</strong><span>{uploadedFile.status === "processing" ? "Extracting knowledge…" : "Attached · ready as knowledge source"}</span></div><button type="button" onClick={onClearUpload} title="Remove attachment"><X size={14}/></button></motion.div>}
      <div className="composerTop">
        <textarea value={value} onChange={e=>setValue(e.target.value)} onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();submit();}}} placeholder="Ask Axiom anything… " rows="1" />
        <button className="sendBtn" type="button" onClick={submit} disabled={busy || !value.trim()}>{busy?<LoaderCircle className="spin" size={18}/>:<Send size={17}/>}</button>
      </div>
      <div className="composerBottom">
        <div className="composerChips">
          <button type="button" className="chip uploadChip" onClick={onUpload} title="Add knowledge"><Plus size={13}/></button>
          <span className="chip sourceChip"><FileText size={12}/>{documentLabel(sourceMode,rowIndex,uploadedFile)}</span>
          <span className="chip providerChip"><ProviderBadge provider={llmProvider}/>{axiomMode==="local"?"Local":providers[llmProvider].name}</span>
          {(axiomMode==="grounded" || axiomMode==="hybrid") && <span className="chip"><Layers3 size={12}/>Top {topK}</span>}
        </div>
        <span className="keyHint">Enter to send · Shift + Enter for new line</span>
      </div>
    </div>
  );
}


function AuthPanel({ open, onClose, onGuest, onAuthenticated, initialMode = "welcome" }) {
  const [mode, setMode] = useState(initialMode);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => { if (open) { setMode(initialMode); setError(""); setPassword(""); } }, [open, initialMode]);
  if (!open) return null;

  const submit = async (e) => {
    e.preventDefault();
    setError(""); setBusy(true);
    try {
      const endpoint = mode === "register" ? "register" : "login";
      const payload = mode === "register" ? { name, email, password } : { email, password };
      const r = await fetch(`${API}/api/auth/${endpoint}`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload) });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || "Authentication failed");
      localStorage.setItem("trustrag-token", d.token);
      localStorage.setItem("trustrag-access-mode", "account");
      onAuthenticated(d.user, d.token);
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  };

  return <div className="authBackdrop">
    <motion.div className="authCard" initial={{opacity:0,scale:.97,y:10}} animate={{opacity:1,scale:1,y:0}}>
      {onClose && <button className="authClose" onClick={onClose}><X size={17}/></button>}
      <LogoMark/>
      {mode === "welcome" ? <>
        <h2>Welcome to TrustRAG</h2>
        <p>Use TrustRAG instantly as a guest, or create an account for a persistent identity.</p>
        <div className="authChoices">
          <button className="authPrimary" onClick={()=>setMode("register")}><UserPlus size={16}/> Create account <ArrowRight size={15}/></button>
          <button className="authSecondary" onClick={()=>setMode("login")}><LogIn size={16}/> Sign in</button>
          <button className="guestButton" onClick={onGuest}>Continue as guest</button>
        </div>
        <small className="authNote">Your LLM API key is still sent only with the current request.</small>
      </> : <>
        <span className="authEyebrow">{mode === "register" ? "CREATE ACCOUNT" : "WELCOME BACK"}</span>
        <h2>{mode === "register" ? "Create your TrustRAG account" : "Sign in to TrustRAG"}</h2>
        <form className="authForm" onSubmit={submit}>
          {mode === "register" && <label>Name<input value={name} onChange={e=>setName(e.target.value)} placeholder="Your name" minLength="2" required/></label>}
          <label>Email<div className="authInput"><Mail size={15}/><input type="email" value={email} onChange={e=>setEmail(e.target.value)} placeholder="you@example.com" required/></div></label>
          <label>Password<div className="authInput"><LockKeyhole size={15}/><input type="password" value={password} onChange={e=>setPassword(e.target.value)} placeholder={mode === "register" ? "At least 8 characters" : "Your password"} minLength={mode === "register" ? 8 : 1} required/></div></label>
          {error && <div className="authError"><CircleAlert size={14}/>{error}</div>}
          <button className="authPrimary" disabled={busy}>{busy?<LoaderCircle className="spin" size={16}/>:mode === "register"?<UserPlus size={16}/>:<LogIn size={16}/>} {mode === "register" ? "Create account" : "Sign in"}</button>
        </form>
        <button className="authSwitch" onClick={()=>setMode(mode === "register" ? "login" : "register")}>{mode === "register" ? "Already have an account? Sign in" : "New to TrustRAG? Create account"}</button>
        <button className="guestButton" onClick={onGuest}>Continue as guest</button>
      </>}
    </motion.div>
  </div>;
}

function AccountButton({ user, guest, onSignIn, onLogout }) {
  const [open, setOpen] = useState(false);
  if (guest || !user) return <button className="accountButton guest" onClick={onSignIn}><UserRound size={15}/> Guest <span>Sign in</span></button>;
  const initials = user.name.split(/\s+/).map(x=>x[0]).join("").slice(0,2).toUpperCase();
  return <div className="accountWrap">
    <button className="accountButton" onClick={()=>setOpen(!open)}><span className="accountAvatar">{initials}</span><span>{user.name}</span><ChevronDown size={13}/></button>
    {open && <div className="accountMenu"><strong>{user.name}</strong><span>{user.email}</span><button onClick={()=>{setOpen(false);onLogout();}}><LogOut size={14}/> Sign out</button></div>}
  </div>;
}

function AxiomMenu({ mode, setMode, busy }) {
  const [open, setOpen] = useState(false);
  const items = [
    ["auto","Axiom Auto","Chooses the best response path"],
    ["grounded","Grounded","Uses only attached knowledge"],
    ["hybrid","Hybrid","Evidence plus clearly separated reasoning"],
    ["general","General","General model knowledge"],
    ["local","Local","Runs through your local Ollama model"],
  ];
  const current = items.find(x=>x[0]===mode) || items[0];
  return <div className="axiomMenuWrap">
    <button className="axiomMenuButton" onClick={()=>setOpen(!open)}><LogoMark compact active={busy}/><span><strong>Axiom</strong><small>{current[1].replace("Axiom ","")}</small></span><ChevronDown size={14}/></button>
    <AnimatePresence>{open && <motion.div className="axiomMenu" initial={{opacity:0,y:-5,scale:.98}} animate={{opacity:1,y:0,scale:1}} exit={{opacity:0,y:-4,scale:.98}} transition={{duration:.14}}>
      <div className="axiomMenuTitle">Response mode</div>
      {items.map(([id,label,desc])=><button key={id} className={mode===id?"active":""} onClick={()=>{setMode(id);setOpen(false)}}><span className="menuCheck">{mode===id?<CheckCircle2 size={15}/>:<span/>}</span><span><strong>{label}</strong><small>{desc}</small></span></button>)}
    </motion.div>}</AnimatePresence>
  </div>;
}



const AXIOM_MODES = [
  ["auto", "Axiom Auto", "Best path automatically"],
  ["grounded", "Grounded", "Only your provided evidence"],
  ["hybrid", "Hybrid", "Evidence + separated reasoning"],
  ["general", "General", "General model knowledge"],
  ["local", "Local", "Your local Ollama model"],
];

function ModeSelect({ mode, setMode, busy, placement = "top" }) {
  const [open, setOpen] = useState(false);
  const current = AXIOM_MODES.find(x => x[0] === mode) || AXIOM_MODES[0];
  return <div className={`modeSelect modeSelect-${placement}`}>
    <button className="modeSelectButton" type="button" onClick={()=>setOpen(v=>!v)}>
      <LogoMark compact active={busy}/>
      <span>{placement === "composer" ? current[1] : current[1].replace("Axiom ", "")}</span>
      <ChevronDown size={14}/>
    </button>
    <AnimatePresence>{open && <motion.div className={`modeDropdown ${placement}`} initial={{opacity:0,y:6,scale:.985}} animate={{opacity:1,y:0,scale:1}} exit={{opacity:0,y:4,scale:.985}} transition={{duration:.14}}>
      <div className="modeDropdownLabel">Axiom response mode</div>
      {AXIOM_MODES.map(([id,label,desc]) => <button key={id} type="button" className={mode===id?"active":""} onClick={()=>{setMode(id);setOpen(false)}}>
        <span className="modeDot">{mode===id?<CheckCircle2 size={14}/>:<span/>}</span>
        <span><strong>{label}</strong><small>{desc}</small></span>
      </button>)}
    </motion.div>}</AnimatePresence>
  </div>;
}

function ProSidebar({
  open, onToggle, onNewChat, chats, currentChatId, onOpenChat, onDeleteChat, onRenameChat,
  user, guest, onSignIn, onLogout, onKnowledge, onApiKeys, activeKeyCount
}) {
  if (!open) return <button className="floatingSidebarToggle" onClick={onToggle} title="Open sidebar"><PanelLeftOpen size={18}/></button>;
  const today = chats.slice(0,5);
  return <motion.aside className="proSidebar" initial={{x:-14,opacity:0}} animate={{x:0,opacity:1}} transition={{duration:.18}}>
    <div className="proBrandRow">
      <Brand/>
      <button className="cleanIconBtn" onClick={onToggle} title="Collapse sidebar"><PanelLeftClose size={17}/></button>
    </div>
    <button className="primaryNewChat" onClick={onNewChat}><Plus size={17}/> New Chat <span>⌘K</span></button>
    <nav className="primaryNav">
      <button className="active"><Home size={16}/> Home</button>
      <button onClick={onKnowledge}><BookOpen size={16}/> Knowledge Bases</button>
    </nav>
    <div className="sidebarSectionTitle">Recent Conversations</div>
    <div className="recentList">
      {today.length === 0 ? <div className="recentEmpty">Your conversations will appear here.</div> : today.map(chat => <div key={chat.id} className={`recentRow ${currentChatId===chat.id?"active":""}`}>
        <button className="recentOpen" onClick={()=>onOpenChat(chat.id)}>{chat.title}</button>
        <div className="recentMore">
          <button onClick={()=>onRenameChat(chat)} title="Rename"><Pencil size={12}/></button>
          <button onClick={()=>onDeleteChat(chat)} title="Delete"><Trash2 size={12}/></button>
        </div>
      </div>)}
    </div>
    <div className="sidebarSpacer"/>
    <button className="apiSummaryCard" onClick={onApiKeys}>
      <div><strong>API Keys</strong><small>{activeKeyCount} {activeKeyCount===1?"key":"keys"} active</small></div>
      <span className={activeKeyCount?"connected":""}>{activeKeyCount?"Connected":"Set up"}</span>
      <ChevronRight size={15}/>
    </button>
    <div className="sidebarAccount">
      {guest || !user ? <button onClick={onSignIn}><span className="miniAvatar"><UserRound size={15}/></span><span><strong>Guest</strong><small>Sign in to sync your data</small></span><ChevronRight size={15}/></button> : <button onClick={onLogout}><span className="miniAvatar initials">{user.name.split(/\s+/).map(x=>x[0]).join("").slice(0,2).toUpperCase()}</span><span><strong>{user.name}</strong><small>{user.email}</small></span><LogOut size={14}/></button>}
    </div>
  </motion.aside>;
}

function KnowledgeModal({ open, onClose, sourceMode, setSourceMode, rowIndex, setRowIndex, loadDatasetRow, datasetInfo, documentText, setDocumentText, uploadedFile, onUploadFile, onClearFile, topK, setTopK, uploadError }) {
  if (!open) return null;
  return <div className="modalBackdrop" onMouseDown={e=>{if(e.target===e.currentTarget) onClose()}}>
    <motion.div className="settingsModal knowledgeModal" initial={{opacity:0,y:12,scale:.985}} animate={{opacity:1,y:0,scale:1}}>
      <div className="modalHeader"><div><span className="modalEyebrow">Knowledge</span><h2>Knowledge Base</h2><p>Choose what Axiom is allowed to retrieve from.</p></div><button className="cleanIconBtn" onClick={onClose}><X size={18}/></button></div>
      <div className="sourceTabs proTabs">
        <button className={sourceMode==="dataset"?"active":""} onClick={()=>setSourceMode("dataset")}><Database size={14}/> Dataset</button>
        <button className={sourceMode==="paste"?"active":""} onClick={()=>setSourceMode("paste")}><FileText size={14}/> Text</button>
        <button className={sourceMode==="file"?"active":""} onClick={()=>setSourceMode("file")}><Upload size={14}/> File</button>
      </div>
      {sourceMode === "dataset" && <div className="modalFieldGroup"><label>FACTS dataset row</label><div className="datasetRow"><input type="number" min="0" max={datasetInfo?.max_index ?? 859} value={rowIndex} onChange={e=>setRowIndex(Number(e.target.value))}/><button onClick={loadDatasetRow}>Load row</button></div><small>Available rows: 0–{datasetInfo?.max_index ?? 859}</small></div>}
      {sourceMode === "paste" && <div className="modalFieldGroup"><label>Source text</label><textarea value={documentText} onChange={e=>setDocumentText(e.target.value)} placeholder="Paste the evidence Axiom should use…"/></div>}
      {sourceMode === "file" && <div className="modalFieldGroup"><label>Attached source</label>{uploadedFile?<div className="modalAttachment"><AttachmentVisual attachment={uploadedFile}/><div><strong>{uploadedFile.name}</strong><small>{uploadedFile.status === "processing" ? "Extracting…" : `${uploadedFile.characters?.toLocaleString() || 0} characters extracted`}</small></div><button onClick={onClearFile}><Trash2 size={14}/></button></div>:<button className="largeUploadButton" onClick={onUploadFile}><Upload size={18}/><span><strong>Upload a file</strong><small>PDF, text, image, audio or video</small></span></button>}{uploadError&&<div className="inlineError"><CircleAlert size={14}/>{uploadError}</div>}</div>}
      <div className="retrievalControl"><div><strong>Retrieval depth</strong><small>How many evidence chunks Axiom retrieves.</small></div><span>{topK}</span><input type="range" min="1" max="8" value={topK} onChange={e=>setTopK(Number(e.target.value))}/></div>
      <div className="modalActions"><button className="secondaryButton" onClick={onClose}>Done</button></div>
    </motion.div>
  </div>;
}

function ApiKeysModal({ open, onClose, llm, setLlm }) {
  if (!open) return null;
  const changeProvider = (provider) => {
    const next = providers[provider];
    setLlm({ provider, api_key: readSessionKey(provider), model: next.model, base_url: next.base });
  };
  const active = Object.keys(providers).filter(id => id !== "ollama" && id !== "custom" && readSessionKey(id)).length;
  return <div className="modalBackdrop" onMouseDown={e=>{if(e.target===e.currentTarget) onClose()}}>
    <motion.div className="settingsModal" initial={{opacity:0,y:12,scale:.985}} animate={{opacity:1,y:0,scale:1}}>
      <div className="modalHeader"><div><span className="modalEyebrow">Connection</span><h2>API Keys & Model</h2><p>Keys stay hidden after save and are kept only in this browser tab.</p></div><button className="cleanIconBtn" onClick={onClose}><X size={18}/></button></div>
      <div className="connectedBanner"><ShieldCheck size={17}/><div><strong>{active ? `${active} provider${active===1?"":"s"} connected` : "No API keys saved"}</strong><small>Axiom sends the selected key only with the current model request.</small></div></div>
      <div className="modalFieldGroup"><label>Provider</label><select value={llm.provider} onChange={e=>changeProvider(e.target.value)}>{Object.entries(providers).map(([id,p])=><option key={id} value={id}>{p.name}</option>)}</select></div>
      {llm.provider !== "ollama" && <div className="modalFieldGroup"><label>API key</label><SecureKeyField provider={llm.provider} value={llm.api_key} onChange={api_key=>setLlm({...llm,api_key})}/></div>}
      <div className="modalFieldGroup"><label>Model</label><input value={llm.model} onChange={e=>setLlm({...llm,model:e.target.value})} placeholder="Model name"/></div>
      {(llm.provider === "custom" || llm.provider === "ollama") && <div className="modalFieldGroup"><label>Base URL</label><input value={llm.base_url} onChange={e=>setLlm({...llm,base_url:e.target.value})}/></div>}
      <div className="modalActions"><button className="secondaryButton" onClick={onClose}>Done</button></div>
    </motion.div>
  </div>;
}

function ResultSourceCard({ item, index }) {
  return <div className="resultSourceCard"><div className="sourceFileIcon"><FileText size={15}/></div><div><strong>Source {index+1}</strong><small>{item.text?.slice(0,115) || "Retrieved evidence"}{item.text?.length>115?"…":""}</small></div><span>{Math.round(Math.max(0,Math.min(1,item.score||0))*100)}%</span></div>;
}

function GroundcheckRail({ result, onClose, visible }) {
  const [tab, setTab] = useState("sources");
  if (!visible) return null;
  const gc = result?.groundcheck;
  const pct = gc ? Math.round(gc.faithfulness_score*100) : null;
  return <motion.aside className="groundRail" initial={{x:16,opacity:0}} animate={{x:0,opacity:1}} transition={{duration:.18}}>
    <div className="railHeader"><div><strong><ShieldCheck size={16}/> GroundCheck</strong><small>Verify responses and explore sources.</small></div><button className="cleanIconBtn" onClick={onClose}><PanelRightClose size={17}/></button></div>
    <div className="railTabs"><button className={tab==="sources"?"active":""} onClick={()=>setTab("sources")}>Sources</button><button className={tab==="details"?"active":""} onClick={()=>setTab("details")}>Details</button></div>
    {tab === "sources" ? <>
      <div className="railSection"><div className="railLabel">Linked Sources {result?.retrieved?.length ? `(${result.retrieved.length})` : ""}</div>{result?.retrieved?.length ? <div className="sourceCardList">{result.retrieved.slice(0,3).map((r,i)=><ResultSourceCard key={i} item={r} index={i}/>)}</div> : <div className="railEmpty">Sources will appear here after a grounded or hybrid answer.</div>}{result?.retrieved?.length>3&&<button className="viewSources">View all sources <ArrowRight size={13}/></button>}</div>
      <div className="railDivider"/>
      <div className="railSection"><div className="railLabel">Verification</div>{gc?<div className={`verificationSummary ${pct>=80?"good":pct>=55?"warn":"bad"}`}><span className="verificationIcon"><ShieldCheck size={18}/></span><span><strong>{gc.overall_verdict === "FAITHFUL" ? "Grounded" : gc.overall_verdict.replaceAll("_"," ")}</strong><small>{pct}% faithful · {gc.claims?.filter(c=>c.verdict==="SUPPORTED").length || 0} supported claims</small></span><ChevronRight size={14}/></div>:<div className="railEmpty">GroundCheck results will appear after an evidence-backed response.</div>}</div>
    </> : <div className="railSection detailRail">
      <div className="railLabel">GroundCheck Details</div>
      {gc ? <>
        <div className="detailScore"><strong>{pct}%</strong><span>faithfulness</span></div>
        <div className="detailStats"><span>Supported <b>{gc.claims?.filter(c=>c.verdict==="SUPPORTED").length || 0}</b></span><span>Unsupported <b>{gc.claims?.filter(c=>c.verdict==="UNSUPPORTED").length || 0}</b></span><span>Contradictions <b>{gc.claims?.filter(c=>c.verdict==="CONTRADICTION").length || 0}</b></span></div>
        <div className="claimDetailList">{(gc.claims || []).map((c,i)=><div className={`claimDetail ${c.verdict?.toLowerCase()}`} key={i}><strong>Claim {i+1} · {c.verdict}</strong><p>{c.claim}</p>{c.explanation&&<small>{c.explanation}</small>}</div>)}</div>
      </> : <div className="railEmpty">No verification details yet.</div>}
    </div>}
    <div className="railDivider"/>
    <div className="railSection"><div className="railLabel">Model Utilities</div><div className="utilityList"><button disabled title="Available in a later TrustRAG capability"><RefreshCcw size={14}/> Re-run with different model</button><button disabled title="Available in a later TrustRAG capability"><Link2 size={14}/> Compare responses</button><button disabled title="Available in a later TrustRAG capability"><FileCheck2 size={14}/> Extract key facts</button><button disabled title="Available in a later TrustRAG capability"><FileText size={14}/> Generate executive brief</button></div></div>
  </motion.aside>;
}

function ProMessage({ msg, onInspect }) {
  const ai = msg.role === "assistant";
  return <motion.div className={`proMessage ${ai?"ai":"human"}`} initial={{opacity:0,y:8}} animate={{opacity:1,y:0}} transition={{duration:.18}}>
    <div className="proAvatar">{ai?<LogoMark compact/>:<UserRound size={16}/>}</div>
    <div className={`proBubble ${ai?"ai":"human"}`}>
      {!ai && msg.attachment && <div className="messageAttachmentCard"><AttachmentVisual attachment={msg.attachment}/><div><strong>{msg.attachment.name}</strong><small>{msg.attachment.characters?`${msg.attachment.characters.toLocaleString()} characters extracted`:"Attached source"}</small></div></div>}
      <div className={ai?"markdown-body":""}>{ai?<ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>:msg.text}</div>
      {ai && <div className="answerFooter"><div>{msg.result?.groundcheck?<button className="groundedTag" onClick={()=>onInspect(msg.result)}><CheckCircle2 size={13}/> Grounded in {Math.max(1,msg.result.retrieved?.length||0)} source{(msg.result.retrieved?.length||0)===1?"":"s"}</button>:<span className="generalTag">{(msg.result?.mode||"general").toUpperCase()}</span>}</div><div className="responseActions"><button onClick={()=>navigator.clipboard.writeText(msg.text)} title="Copy"><Copy size={14}/></button><button title="Helpful"><ThumbsUp size={14}/></button><button title="Not helpful"><ThumbsDown size={14}/></button></div></div>}
    </div>
  </motion.div>;
}

function ComposerModeButton({ mode, setMode, busy }) { return <ModeSelect mode={mode} setMode={setMode} busy={busy} placement="composer"/>; }

function ProComposer({ busy, value, setValue, onSubmit, onUpload, onKnowledge, uploadedFile, attachmentPending, onClearUpload, axiomMode, setAxiomMode }) {
  const [plusOpen,setPlusOpen] = useState(false);
  return <div className="proComposerWrap">
    <AnimatePresence>{uploadedFile && attachmentPending && <motion.div className="composerAttachmentCard" initial={{opacity:0,y:6}} animate={{opacity:1,y:0}}><AttachmentVisual attachment={uploadedFile}/><div><strong>{uploadedFile.name}</strong><small>{uploadedFile.status === "processing"?"Extracting knowledge…":"Ready to use as evidence"}</small></div><button onClick={onClearUpload}><X size={14}/></button></motion.div>}</AnimatePresence>
    <div className="proComposer">
      <textarea rows="1" value={value} onChange={e=>setValue(e.target.value)} onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();onSubmit()}}} placeholder="Ask anything…"/>
      <div className="composerToolbar">
        <div className="composerLeftTools">
          <div className="plusMenuWrap"><button className="composerIconBtn" onClick={()=>setPlusOpen(v=>!v)}><Plus size={18}/></button><AnimatePresence>{plusOpen&&<motion.div className="plusMenu" initial={{opacity:0,y:6,scale:.985}} animate={{opacity:1,y:0,scale:1}} exit={{opacity:0,y:4,scale:.985}}><button onClick={()=>{setPlusOpen(false);onUpload()}}><Upload size={15}/> Upload File</button><button onClick={()=>{setPlusOpen(false);onKnowledge()}}><BookOpen size={15}/> Knowledge Base</button></motion.div>}</AnimatePresence></div>
          <button className="composerIconBtn" onClick={onUpload} title="Attach file"><Paperclip size={16}/></button>
          <button className="composerIconBtn" onClick={onKnowledge} title="Knowledge base"><Grid2X2 size={15}/></button>
        </div>
        <div className="composerRightTools"><ComposerModeButton mode={axiomMode} setMode={setAxiomMode} busy={busy}/><button className="proSend" disabled={busy || !value.trim()} onClick={onSubmit}>{busy?<LoaderCircle className="spin" size={18}/>:<Send size={17}/>}</button></div>
      </div>
    </div>
    <div className="composerDisclaimer">TrustRAG can make mistakes. Verify important information.</div>
  </div>;
}

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [verify, setVerify] = useState(null);
  const [llm, setLlm] = useState({provider:"gemini", api_key:readSessionKey("gemini"), model:providers.gemini.model, base_url:providers.gemini.base});
  const [topK, setTopK] = useState(4);
  const [axiomMode, setAxiomMode] = useState("auto");
  const [sourceMode, setSourceMode] = useState("dataset");
  const [rowIndex, setRowIndex] = useState(0);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [documentText, setDocumentText] = useState("");
  const [suggestedQuestion, setSuggestedQuestion] = useState("");
  const [uploadedFile, setUploadedFile] = useState(null);
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [theme, setThemeState] = useState("light");
  const [user, setUser] = useState(null);
  const [accessMode, setAccessMode] = useState(() => localStorage.getItem("trustrag-access-mode"));
  const [authOpen, setAuthOpen] = useState(false);
  const [authReady, setAuthReady] = useState(false);
  const [chats, setChats] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [historySearch, setHistorySearch] = useState("");
  const [uploadError, setUploadError] = useState("");
  const [attachmentPending, setAttachmentPending] = useState(false);
  const [knowledgeOpen, setKnowledgeOpen] = useState(false);
  const [apiKeysOpen, setApiKeysOpen] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [composerValue, setComposerValue] = useState("");
  const endRef = useRef(null);
  const globalFileRef = useRef(null);

  const setTheme = (next) => { setThemeState(next); localStorage.setItem("trustrag-theme", next); };

  useEffect(() => {
    const token = localStorage.getItem("trustrag-token");
    if (!token) { setAuthReady(true); return; }
    fetch(`${API}/api/auth/me`, { headers: { Authorization: `Bearer ${token}` } })
      .then(async r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(d => { setUser(d.user); setAccessMode("account"); })
      .catch(() => { localStorage.removeItem("trustrag-token"); localStorage.removeItem("trustrag-access-mode"); setAccessMode(null); })
      .finally(() => setAuthReady(true));
  }, []);

  const continueGuest = () => { localStorage.setItem("trustrag-access-mode", "guest"); setAccessMode("guest"); setUser(null); setChats([]); setCurrentChatId(null); setAuthOpen(false); };
  const authenticated = (nextUser) => { setUser(nextUser); setAccessMode("account"); setAuthOpen(false); };
  const logout = async () => {
    const token = localStorage.getItem("trustrag-token");
    try {
      if (token) await fetch(`${API}/api/auth/logout`, { method:"POST", headers:{Authorization:`Bearer ${token}`} });
    } catch (_) {}
    localStorage.removeItem("trustrag-token");
    localStorage.removeItem("trustrag-access-mode");
    setUser(null);
    setAccessMode(null);
    setChats([]);
    setCurrentChatId(null);
    setMessages([]);
    setVerify(null);
    setAuthOpen(false);
  };

  const authHeaders = () => {
    const token = localStorage.getItem("trustrag-token");
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  async function refreshChats() {
    if (!user) { setChats([]); return; }
    try {
      const r = await fetch(`${API}/api/chats`, { headers: authHeaders() });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || "Could not load chat history");
      setChats(d.chats || []);
    } catch (e) { setError(e.message); }
  }

  useEffect(() => { if (user) refreshChats(); else setChats([]); }, [user]);

  const chatPayload = (title, chatMessages) => ({
    title, messages: chatMessages.map(m => m.attachment ? {...m, attachment:{...m.attachment, preview_url:null}} : m), source_mode: sourceMode,
    source_name: uploadedFile?.name || (sourceMode === "dataset" ? `FACTS #${rowIndex}` : sourceMode === "paste" ? "Pasted text" : ""),
    document_text: documentText, row_index: rowIndex, provider: llm.provider,
    model: llm.model, base_url: llm.base_url, top_k: topK
  });

  async function persistChat(chatMessages, title) {
    if (!user || chatMessages.length === 0) return currentChatId;
    const existingTitle = currentChatId ? chats.find(c=>c.id === currentChatId)?.title : "";
    const finalTitle = title || existingTitle || chatMessages.find(m=>m.role === "user")?.text?.slice(0, 58) || "New chat";
    const id = currentChatId;
    const r = await fetch(id ? `${API}/api/chats/${id}` : `${API}/api/chats`, {
      method: id ? "PUT" : "POST",
      headers: { "Content-Type":"application/json", ...authHeaders() },
      body: JSON.stringify(chatPayload(finalTitle, chatMessages))
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "Could not save chat");
    const savedId = d.chat.id;
    if (!id) setCurrentChatId(savedId);
    await refreshChats();
    return savedId;
  }

  async function openChat(chatId) {
    try {
      const r = await fetch(`${API}/api/chats/${chatId}`, { headers: authHeaders() });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || "Could not open chat");
      const chat = d.chat;
      setCurrentChatId(chat.id); setMessages(chat.messages || []); setVerify(null); setError("");
      setDocumentText(chat.document_text || ""); setSourceMode(chat.source_mode || "dataset");
      setRowIndex(chat.row_index || 0); setTopK(chat.top_k || 4);
      if (providers[chat.provider]) setLlm(prev => ({...prev, provider:chat.provider, model:chat.model || providers[chat.provider].model, base_url:chat.base_url || providers[chat.provider].base}));
      setUploadedFile(chat.source_mode === "file" && chat.source_name ? {name:chat.source_name,size:0,type:"",kind:"file",preview_url:null,status:"loaded",characters:(chat.document_text || "").length} : null);
      setSuggestedQuestion("");
    } catch(e) { setError(e.message); }
  }

  async function deleteChat(chat) {
    if (!window.confirm(`Delete “${chat.title}”?`)) return;
    try {
      const r = await fetch(`${API}/api/chats/${chat.id}`, { method:"DELETE", headers:authHeaders() });
      const d = await r.json(); if (!r.ok) throw new Error(d.detail || "Could not delete chat");
      if (currentChatId === chat.id) { setCurrentChatId(null); setMessages([]); setVerify(null); }
      await refreshChats();
    } catch(e) { setError(e.message); }
  }

  async function renameChat(chat) {
    const title = window.prompt("Rename chat", chat.title);
    if (!title || title.trim() === chat.title) return;
    try {
      const r = await fetch(`${API}/api/chats/${chat.id}/title`, { method:"PATCH", headers:{"Content-Type":"application/json", ...authHeaders()}, body:JSON.stringify({title:title.trim()}) });
      const d = await r.json(); if (!r.ok) throw new Error(d.detail || "Could not rename chat");
      await refreshChats();
    } catch(e) { setError(e.message); }
  }

  async function processUpload(file) {
    if (!file) return;
    if (uploadedFile?.preview_url) URL.revokeObjectURL(uploadedFile.preview_url);
    const type = file.type || "application/octet-stream";
    const kind = attachmentKind(type);
    const preview_url = kind === "image" ? URL.createObjectURL(file) : null;
    setUploadError(""); setError(""); setSourceMode("file"); setSuggestedQuestion("");
    setUploadedFile({ name:file.name, size:file.size, type, kind, preview_url, status:"processing" });
    setAttachmentPending(true);
    try {
      const form = new FormData(); form.append("file", file); form.append("provider", llm.provider); form.append("api_key", llm.api_key); form.append("model", llm.model);
      const r = await fetch(`${API}/api/media/extract`, { method:"POST", body:form });
      const d = await r.json(); if (!r.ok) throw new Error(d.detail || "Could not process this file");
      setDocumentText(d.document_text);
      setUploadedFile({name:d.filename || file.name,size:d.size || file.size,type:d.mime_type || type,kind,preview_url,status:"loaded",characters:d.characters,extraction:d.extraction});
      setAttachmentPending(true);
    } catch(e) {
      if (preview_url) URL.revokeObjectURL(preview_url);
      setDocumentText(""); setUploadedFile(null); setAttachmentPending(false); setUploadError(e.message); setError(e.message);
    }
    finally { if (globalFileRef.current) globalFileRef.current.value = ""; }
  }

  const clearUploadedFile = () => {
    if (uploadedFile?.preview_url) URL.revokeObjectURL(uploadedFile.preview_url);
    setUploadedFile(null); setDocumentText(""); setUploadError(""); setAttachmentPending(false);
  };

  useEffect(()=>{
    fetch(`${API}/api/dataset/info`).then(r=>r.json()).then(setDatasetInfo).catch(()=>{});
    loadRow(0);
  },[]);

  useEffect(()=>{ document.documentElement.dataset.theme = theme; },[theme]);
  useEffect(()=>{ endRef.current?.scrollIntoView({behavior:"smooth", block:"end"}); },[messages,busy]);

  async function loadRow(idx=rowIndex){
    setError("");
    setUploadedFile(null);
    setAttachmentPending(false);
    try {
      const r=await fetch(`${API}/api/dataset/${idx}`);
      const d=await r.json();
      if(!r.ok) throw new Error(d.detail||"Could not load row");
      setDocumentText(d.document_text);
      setSuggestedQuestion(d.suggested_question);
    } catch(e){ setError(e.message); }
  }

  async function ask(q){
    if(!q || busy) return;
    setError("");
    setSuggestedQuestion("");
    const attachment = attachmentPending && uploadedFile ? {name:uploadedFile.name, characters:uploadedFile.characters || 0, type:uploadedFile.type || "", kind:uploadedFile.kind || attachmentKind(uploadedFile.type || ""), preview_url:uploadedFile.preview_url || null} : null;
    setMessages(m=>[...m,{role:"user",text:q,attachment}]);
    if (attachment) setAttachmentPending(false);
    setBusy(true);
    try{
      const r=await fetch(`${API}/api/ask`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({document_text:documentText,question:q,top_k:topK,axiom_mode:axiomMode,source_name:uploadedFile?.name || (sourceMode==="dataset"?`FACTS #${rowIndex}`:"Knowledge source"),llm})});
      const d=await r.json();
      if(!r.ok) throw new Error(d.detail||"Request failed");
      const assistantMessage = {role:"assistant",text:d.answer,result:d};
      setMessages(m=>{ const next=[...m,assistantMessage]; if(user) persistChat(next).catch(e=>setError(e.message)); return next; });
    }catch(e){ setError(e.message); }
    finally{setBusy(false);}
  }

  const hasChat = messages.length > 0;
  const latestResult = [...messages].reverse().find(m => m.role === "assistant" && m.result)?.result || null;
  const activeResult = verify || latestResult;
  const activeKeyCount = Object.keys(providers).filter(id => id !== "ollama" && id !== "custom" && readSessionKey(id)).length;
  const activeChatTitle = currentChatId
    ? (chats.find(c=>c.id===currentChatId)?.title || "Conversation")
    : (messages.find(m=>m.role==="user")?.text?.slice(0,60) || "New Chat");

  const resetChat = () => {
    setMessages([]); setVerify(null); setError(""); setCurrentChatId(null);
    setSuggestedQuestion(""); setComposerValue(""); setInspectorOpen(true);
  };

  const submitComposer = () => {
    const q = composerValue.trim();
    if (!q || busy) return;
    setComposerValue("");
    ask(q);
  };

  return (
    <div className={`professionalApp ${sidebarOpen?"sidebarOpen":"sidebarClosed"} ${inspectorOpen?"inspectorOpen":"inspectorClosed"}`}>
      <input ref={globalFileRef} type="file" accept=".txt,.md,.csv,.json,.pdf,image/*,audio/*,video/*" hidden onChange={e=>processUpload(e.target.files?.[0])}/>

      <ProSidebar
        open={sidebarOpen}
        onToggle={()=>setSidebarOpen(v=>!v)}
        onNewChat={resetChat}
        chats={chats}
        currentChatId={currentChatId}
        onOpenChat={openChat}
        onDeleteChat={deleteChat}
        onRenameChat={renameChat}
        user={user}
        guest={accessMode === "guest"}
        onSignIn={()=>setAuthOpen(true)}
        onLogout={logout}
        onKnowledge={()=>setKnowledgeOpen(true)}
        onApiKeys={()=>setApiKeysOpen(true)}
        activeKeyCount={activeKeyCount}
      />

      <main className="proWorkspace">
        <header className="proTopbar">
          <div className="chatTitleGroup">
            {!sidebarOpen && <button className="cleanIconBtn inlineSidebarToggle" onClick={()=>setSidebarOpen(true)} title="Open sidebar"><PanelLeftOpen size={18}/></button>}
            {hasChat && <div className="conversationTitle" title={activeChatTitle}>{activeChatTitle}</div>}
          </div>
          <div className="topbarTools">
            <button className={`groundcheckToggle ${inspectorOpen?"active":""}`} onClick={()=>setInspectorOpen(v=>!v)}><ShieldCheck size={15}/><span>GroundCheck</span>{inspectorOpen?<PanelRightClose size={14}/>:<PanelRightOpen size={14}/>}</button>
            <button className="cleanIconBtn" onClick={()=>setTheme(theme==="light"?"dark":"light")} title="Toggle theme">{theme==="light"?<Moon size={16}/>:<Sun size={16}/>}</button>
          </div>
        </header>

        <div className="workspaceBody">
          <section className="conversationPane">
            <div className="conversationScroll">
              {!hasChat ? <div className="emptyConversation">
                <LogoMark hero active={busy}/>
                <h1>How can Axiom help?</h1>
                <p>Ask a question, attach evidence, or open a knowledge base. Axiom can answer generally, grounded in your sources, or with hybrid reasoning.</p>
                <div className="quickActions">
                  <button onClick={()=>globalFileRef.current?.click()}><Upload size={16}/><span><strong>Upload a document</strong><small>PDF, image, text and more</small></span></button>
                  <button onClick={()=>setKnowledgeOpen(true)}><BookOpen size={16}/><span><strong>Open Knowledge Base</strong><small>Dataset, pasted text or retrieval settings</small></span></button>
                  <button onClick={()=>setApiKeysOpen(true)}><KeyRound size={16}/><span><strong>Model connection</strong><small>{activeKeyCount?`${activeKeyCount} API key${activeKeyCount===1?"":"s"} ready`:"Connect a provider"}</small></span></button>
                </div>
              </div> : <div className="proThread">
                {messages.map((m,i)=><ProMessage key={`${m.role}-${i}`} msg={m} onInspect={(result)=>{setVerify(result);setInspectorOpen(true)}}/>)}
                {busy && <WorkingTimeline/>}
                <div ref={endRef}/>
              </div>}
            </div>

            <div className="composerZone">
              <AnimatePresence>{error && <motion.div className="proError" initial={{opacity:0,y:5}} animate={{opacity:1,y:0}} exit={{opacity:0}}><CircleAlert size={14}/><span>{error}</span><button onClick={()=>setError("")}><X size={13}/></button></motion.div>}</AnimatePresence>
              <ProComposer
                busy={busy}
                value={composerValue}
                setValue={setComposerValue}
                onSubmit={submitComposer}
                onUpload={()=>globalFileRef.current?.click()}
                onKnowledge={()=>setKnowledgeOpen(true)}
                uploadedFile={uploadedFile}
                attachmentPending={attachmentPending}
                onClearUpload={clearUploadedFile}
                axiomMode={axiomMode}
                setAxiomMode={setAxiomMode}
              />
            </div>
          </section>

          <GroundcheckRail result={activeResult} visible={inspectorOpen} onClose={()=>setInspectorOpen(false)}/>
        </div>
      </main>

      <KnowledgeModal
        open={knowledgeOpen}
        onClose={()=>setKnowledgeOpen(false)}
        sourceMode={sourceMode}
        setSourceMode={setSourceMode}
        rowIndex={rowIndex}
        setRowIndex={setRowIndex}
        loadDatasetRow={()=>loadRow(rowIndex)}
        datasetInfo={datasetInfo}
        documentText={documentText}
        setDocumentText={setDocumentText}
        uploadedFile={uploadedFile}
        onUploadFile={()=>globalFileRef.current?.click()}
        onClearFile={clearUploadedFile}
        topK={topK}
        setTopK={setTopK}
        uploadError={uploadError}
      />
      <ApiKeysModal open={apiKeysOpen} onClose={()=>setApiKeysOpen(false)} llm={llm} setLlm={setLlm}/>

      {authReady && !accessMode && <AuthPanel open={true} onGuest={continueGuest} onAuthenticated={authenticated} initialMode="welcome" />}
      <AuthPanel open={authOpen} onClose={()=>setAuthOpen(false)} onGuest={continueGuest} onAuthenticated={authenticated} initialMode="login" />
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App/>);