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
  arbiter_eval TEXT,  -- Added for v0.6 to store full arbiter evaluation
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

-- v0.4 - Pairwise + Elo Rating Tables
CREATE TABLE IF NOT EXISTS pairwise_judgment (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT REFERENCES run(id) ON DELETE CASCADE,
  decision_type_id INTEGER REFERENCES decision_type(id),
  judge_model_id INTEGER REFERENCES model(id),
  winner_model_id INTEGER REFERENCES model(id),
  loser_model_id INTEGER REFERENCES model(id),
  reason TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(run_id, winner_model_id, loser_model_id)
);

CREATE TABLE IF NOT EXISTS model_rating (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id INTEGER REFERENCES model(id),
  decision_type_id INTEGER REFERENCES decision_type(id),
  algo TEXT NOT NULL DEFAULT 'elo', -- "elo" | "trueskill"
  rating REAL DEFAULT 1200.0,        -- Elo rating (start at 1200)
  k_factor REAL DEFAULT 32.0,        -- Elo K factor
  mu REAL,                           -- TrueSkill mu (future)
  sigma REAL,                        -- TrueSkill sigma (future)
  matches INTEGER DEFAULT 0,
  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(model_id, decision_type_id, algo)
);

-- Performance indexes for v0.4
CREATE INDEX IF NOT EXISTS idx_model_rating_type_rating
  ON model_rating(decision_type_id, algo, rating DESC);

CREATE INDEX IF NOT EXISTS idx_pairwise_type_time
  ON pairwise_judgment(decision_type_id, created_at);

-- Insert default decision types
INSERT OR IGNORE INTO decision_type (key, label) VALUES
  ('project-structure', 'Project Structure'),
  ('database', 'Database Architecture'),
  ('deployment', 'Deployment Strategy'),
  ('api-design', 'API Design'),
  ('authentication', 'Authentication'),
  ('frontend-framework', 'Frontend Framework'),
  ('testing-strategy', 'Testing Strategy');