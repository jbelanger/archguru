-- ArchGuru v0.3 - Minimal SQLite Schema for Persistence Baseline
-- Based on roadmap.md technical implementation guide

-- Core tables for decisions and model responses
CREATE TABLE IF NOT EXISTS model (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT UNIQUE NOT NULL,
  provider TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS decision_type (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  key TEXT UNIQUE NOT NULL,
  label TEXT
);

CREATE TABLE IF NOT EXISTS run (
  id TEXT PRIMARY KEY,
  decision_type_id INTEGER REFERENCES decision_type(id),
  language TEXT,
  framework TEXT,
  requirements TEXT,
  prompt_version TEXT,
  arbiter_model_id INTEGER REFERENCES model(id),
  consensus_reco TEXT,
  debate_summary TEXT,
  total_time_sec REAL,
  error TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_response (
  id TEXT PRIMARY KEY,
  run_id TEXT REFERENCES run(id) ON DELETE CASCADE,
  model_id INTEGER REFERENCES model(id),
  team TEXT,
  recommendation TEXT,
  reasoning TEXT,
  trade_offs TEXT, -- JSON as TEXT for now
  confidence_score REAL,
  response_time_sec REAL,
  success BOOLEAN DEFAULT TRUE,
  error TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tool_call (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  response_id TEXT REFERENCES model_response(id) ON DELETE CASCADE,
  function TEXT,
  arguments TEXT, -- JSON as TEXT for now
  result_excerpt TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default decision types
INSERT OR IGNORE INTO decision_type (key, label) VALUES
  ('project-structure', 'Project Structure'),
  ('database', 'Database Architecture'),
  ('deployment', 'Deployment Strategy'),
  ('api-design', 'API Design'),
  ('authentication', 'Authentication'),
  ('frontend-framework', 'Frontend Framework'),
  ('testing-strategy', 'Testing Strategy');