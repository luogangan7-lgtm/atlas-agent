/**
 * ATLAS Memory v12.0.0 — 多域自主演化知识系统（实体注册 · 关系图 · 多域 · 置信演化 · DeepSeek · MCP:8766）
 *
 * 架构（四层，商业级）：
 *   INJECT  — LRU缓存 + 跳过短/重复 + 时间衰减 + 访问计数 + 注入 memory_type
 *   CAPTURE — agent_end（含用户消息上下文 + 质量过滤≥7 + 冲突检测）
 *             + llm_output 每5轮中途捕获
 *   LEARN   — after_tool_call 拦截搜索工具 → Qdrant
 *   TOOLS   — 9工具 + atlas_merge（近重复合并）+ atlas_obsidian_sync
 *
 * 商业级升级：
 *   ★ omlx Qwen3.5-9B 替代 Ollama qwen2.5（4s vs 25s，质量更高）
 *   ★ 冲突检测（新记忆与旧记忆矛盾时自动 keep_new/keep_old/merge）
 *   ★ 质量评分（提取时评分1-10，只存≥7分）
 *   ★ memory_type 字段（preference/fact/skill/project/constraint/event）
 *   ★ 用户消息纳入 CAPTURE 上下文
 *   ★ 过期记忆自动清理（hit_count=0 + age>90天 + importance=low）
 *   ★ atlas_merge 工具（近重复智能合并）
 *
 * Obsidian Bridge（v9.4.0 新增）：
 *   ★ 单向主权：Qdrant AI记忆 → Atlas_Mirror（只读进化监控台）
 *   ★ 主题聚类导出：memory_type + 主标签分组，[type] topic.md 高辨识度命名
 *   ★ 图谱融合：聚类文件底部生成 [[wikilinks]]，接入 Obsidian 知识图谱
 *   ★ 每日进化日志：_evolution/YYYY-MM-DD.md 按天切分，记录 CAPTURE/MERGE/PRUNE/UPGRADE
 *   ★ Dataview 仪表盘：_index.md 含空数据降级提示
 *
 * v9.5.0 升级：
 *   ★ atlas_feedback — 记忆反馈回路（负评降权/删除，防止错误记忆加权）
 *   ★ atlas_distill  — 知识提炼（DeepSeek云端合成通则，omlx备用）
 *   ★ atlas_timeline — 主题时间线查询
 *   ★ 版本化         — 冲突替换时保留旧版本（status:superseded），不再物理删除
 *   ★ INJECT改进    — 自动过滤负评记忆，优先注入[distilled]通则，追踪注入ID
 *   ★ EVOLVE自动提炼 — 同标签≥5条时自动触发distill，生成通则进入下次检索
 */

import http from 'http';
import https from 'https';
import { createHash } from 'crypto';
import { writeFile, readFile, mkdir, appendFile, unlink, rename, readdir, rmdir } from 'fs/promises';
import { homedir } from 'os';
import { join } from 'path';

// ── 常量 ──────────────────────────────────────────────────────────────────────
const QDRANT               = 'http://127.0.0.1:6333';
const OLLAMA               = 'http://127.0.0.1:11434';
const OMLX                 = 'http://127.0.0.1:7749';
const OMLX_MODEL           = 'Qwen3.5-9B-OptiQ-4bit';
const COLLECTION           = 'atlas_memories_v2';
const EMBED_MODEL          = 'bge-m3';
const VECTOR_DIM           = 1024;
const SCORE_MIN            = 0.45;  // 降低阈值以覆盖L0原始内容（L0约0.50-0.58，L1约0.65+）
const SCORE_DEDUP          = 0.92;
const SCORE_CONFLICT_MIN   = 0.75;  // ★ 冲突检测下界
const SCORE_CONFLICT_MAX   = 0.91;  // ★ 冲突检测上界（DEDUP以上已去重）
const INJECT_LIMIT         = 5;
const MIN_CAPTURE_CHARS    = 200;
const TIMEOUT_MS           = 10_000;
const INJECT_TIMEOUT_MS    = 2_500;  // INJECT hook 专用：超时即降级，不阻塞响应
const EXTRACT_TIMEOUT_MS   = 12_000; // omlx 快得多，12s 足够
const FETCH_TIMEOUT_MS     = 15_000;
const EMBED_CACHE_SIZE     = 200;
const CAPTURE_TURN_INTERVAL = 5;
const CHUNK_SIZE           = 1500;
const CHUNK_OVERLAP        = 300;
const MAX_CHUNKS           = 5;
const DECAY_MAX_PENALTY    = 0.4;
const DECAY_PERIOD_DAYS    = 180;
const BACKUP_DIR           = join(homedir(), '.atlas-backups');
const IMPORTANCE_LEVELS    = ['low', 'medium', 'high', 'critical'];
const HIT_UPGRADE          = { low: 5, medium: 12, high: 25 };
const STALE_AGE_DAYS       = 90;    // ★ 过期记忆清理阈值
const MIN_QUALITY_SCORE    = 7;     // ★ 只存质量≥7的记忆
const SEARCH_TOOL_KEYWORDS = ['search', 'harvester', 'google', 'brave', 'bing', 'duckduckgo', 'serp'];

// ── DeepSeek（云端合成，用于 distill）────────────────────────────────────────
const DEEPSEEK_URL          = 'https://api.deepseek.com';
const DEEPSEEK_MODEL        = 'deepseek-v4-flash';
const DEEPSEEK_TIMEOUT_MS   = 90_000;  // extractL1Content 最多6000 tokens（推理模型约500推理token+5500内容token）
const DEEPSEEK_API_KEY      = process.env.DEEPSEEK_API_KEY ?? '';

// ── 反馈回路 ──────────────────────────────────────────────────────────────────
const FEEDBACK_DECAY        = 0.25;   // 负反馈降幅（wrong/outdated）
const FEEDBACK_BOOST        = 0.05;   // 正反馈升幅（correct）
const FEEDBACK_FILTER_MIN   = 0.5;    // INJECT 过滤门槛（低于此值不注入）
const FEEDBACK_DELETE_FLOOR = 0.2;    // 低于此值直接删除

// ── 知识提炼 ──────────────────────────────────────────────────────────────────
const DISTILL_MIN_COUNT     = 5;      // 同标签最少记忆数才触发提炼
const DISTILL_TAG           = '[distilled]';

// ── 持久化状态 ────────────────────────────────────────────────────────────────
const STATE_FILE = join(process.env.OPENCLAW_STATE_DIR ?? '.', 'atlas-memory-state.json');

// ── Obsidian Bridge 常量 ───────────────────────────────────────────────────────
const OBSIDIAN_VAULT         = process.env.ATLAS_OBSIDIAN_VAULT ?? '';
const OBSIDIAN_MIRROR_DIR    = '_演化';

// ── 分类辅助：读取 taxonomy.json，将 domain → 大类（category）─────────────────
let _taxonomyCache = null;
let _taxonomyMtime = 0;
async function getCategoryForDomain(domain) {
  if (!domain || domain === '未分类') return '其他';
  const primary = domain.split('×')[0].trim();
  if (OBSIDIAN_VAULT) {
    try {
      const taxPath = join(OBSIDIAN_VAULT, 'agent-brain', 'state', 'taxonomy.json');
      const { mtimeMs } = await import('fs').then(m => m.promises.stat(taxPath)).catch(() => ({ mtimeMs: 0 }));
      if (!_taxonomyCache || mtimeMs !== _taxonomyMtime) {
        const raw = await readFile(taxPath, 'utf8').catch(() => null);
        _taxonomyCache = raw ? JSON.parse(raw) : null;
        _taxonomyMtime = mtimeMs;
      }
      if (_taxonomyCache) {
        for (const [catName, catData] of Object.entries(_taxonomyCache.categories ?? {})) {
          const domains = catData.domains ?? [];
          if (domains.includes(primary) || domains.includes(domain)) return catName;
        }
      }
    } catch (_) {}
  }
  return '其他';
}
const EVOLUTION_LOG_SUBDIR   = '';
const MIRROR_LINK_MAX        = 5;
const MIRROR_EXPORT_INTERVAL = 20 * 60 * 1000;         // 加速期：20min（正常 6h）
const IMPORTANCE_ORDER       = { critical: 4, high: 3, medium: 2, low: 1 };

// ── v10 知识库常量 ─────────────────────────────────────────────────────────────
const LEVEL_RAW          = 0;  // L0 原料
const LEVEL_KNOWLEDGE    = 1;  // L1 知识
const LEVEL_INSIGHT      = 2;  // L2 关联
const LEVEL_WISDOM       = 3;  // L3 智识
const LEVEL_META         = 4;  // L4 元规律

const ORGANIZE_INTERVAL_MS   = 3  * 60 * 1000;          // 加速期：3min（正常 1h）
const DOMAIN_INTERVAL_MS     = 5  * 60 * 1000;          // 加速期：5min（正常 2h）
const ASSOCIATE_INTERVAL_MS  = 10 * 60 * 1000;          // 加速期：10min（正常 6h）
const SYNTHESIZE_INTERVAL_MS = 15 * 60 * 1000;          // 加速期：15min（正常 12h）
const META_INTERVAL_MS       = 2 * 60 * 60 * 1000;      // 加速期：2h（正常 24h，避免 L0 生成超过消化速度）
const RESTRUCTURE_INTERVAL_MS   = 2  * 60 * 60 * 1000;  // 加速期：2h（正常 1天）
const COMPLETION_INTERVAL_MS    = 8  * 60 * 1000;       // 加速期：8min（正常 8h）
const MERGE_SIM_THRESHOLD       = 0.88;  // 域质心相似度超过此值 → 合并候选
const SPLIT_COHESION_THRESHOLD  = 0.55;  // 域内聚度低于此值 → 分裂候选
const SPLIT_MIN_NODES           = 15;    // 25→15，批量导入时更快触发分裂
const RESTRUCTURE_MAX_MERGES    = 3;     // 2→3
const RESTRUCTURE_MAX_SPLITS    = 2;     // 1→2
const RESTRUCTURE_MIN_NODES     = 5;     // 参与分析的域最小节点数

const FRESHNESS_REFRESH      = 0.40;  // 低于此值重新验证
const FRESHNESS_INJECT_MIN   = 0.20;  // 低于此值不注入
const DOMAIN_MATCH_SCORE     = 0.80;  // 域精确匹配门槛
const DOMAIN_SUBDOMAIN_SCORE = 0.65;  // 子域匹配门槛
const CLUSTER_MIN_SIZE       = 3;     // 聚类最小记忆数
const CLUSTER_MIN_SCORE      = 0.70;
const ASSOC_MIN_SCORE        = 0.65;  // 关联搜索下界
const ASSOC_MAX_SCORE        = 0.85;  // 关联搜索上界（避开直接重复）
const DECAY_HALF_LIFE        = { fast: 7, medium: 30, slow: 180 }; // 单位：天
const MCP_PORT               = parseInt(process.env.ATLAS_MCP_PORT ?? '8766');
const GITHUB_REPO            = process.env.ATLAS_GITHUB_REPO ?? '';

// ── v11 信息源类型 & TTL ──────────────────────────────────────────────────────
const SOURCE_TYPES = ['trading', 'news', 'social', 'course', 'chat', 'platform_rule', 'process', 'unknown'];

const TTL_MAP = {
  trading:       5 * 60,              // 5分钟（秒）
  news:          3 * 24 * 3600,       // 3天
  social:        7 * 24 * 3600,       // 7天
  chat:          7 * 24 * 3600,       // 7天
  platform_rule: 90 * 24 * 3600,     // 90天
  course:        null,                // 永久
  process:       null,                // 永久
  unknown:       30 * 24 * 3600,     // 30天
};

// 按 source_type 设不同最小捕获字符数
const MIN_CAPTURE_CHARS_MAP = {
  trading:       5,
  news:          30,
  social:        20,
  chat:          10,
  platform_rule: 50,
  course:        150,
  process:       80,
  unknown:       80,
};

// 知识颗粒度方向
const KNOWLEDGE_PURPOSE = {
  UNDERSTANDING: 'understanding',  // 课程学习，深度内化
  PRODUCTION:    'production',     // 内容生产，模板化，可直接使用
  PROCESS:       'process',        // 流程SOP，按步骤执行
};

// 域目录映射（L1/ 下的域文件夹名）
const DOMAIN_DIRS = {
  '情感学': '情感学', '营销': '营销', '战略': '战略',
  '品牌项目': '品牌项目', 'TikTok运营': 'TikTok运营',
  '五金工具-电焊机': '五金工具-电焊机', '储能电池': '储能电池', 'OKX交易': 'OKX交易',
  '其他学习': '其他学习',
  // 旧 domain 名兼容映射
  '01-情感学': '情感学', '02-营销学-刘克亚': '营销', '03-营销学-智多星': '营销',
  '04-战略-刘海峰': '战略', '营销(科特勒)': '营销',
};

// 域描述（用于向量匹配，整理Agent用）
const DOMAIN_DESCRIPTIONS = {
  '营销':        '营销策略 广告文案 品牌传播 流量获取 内容营销 用户转化 市场推广 促销活动 消费者心理',
  '品牌项目':    '香氛品牌 产品开发 品牌策略 视觉设计 产品定位 包装设计 品牌故事 产品规划',
  '情感学':      '情感关系 吸引力 搭讪 社交技巧 约会 恋爱 人际沟通 男女关系 吸引异性',
  '战略':        '商业战略 竞争分析 市场定位 商业模式 长期规划 企业管理 决策框架 战略思维',
  'TikTok运营':  'TikTok短视频 内容创作 算法 运营数据 海外社媒 粉丝增长 账号管理 视频剪辑',
  '五金工具-电焊机': '五金工具 电焊机 工业设备 产品规格 焊接技术 工具选型 供应链 硬件产品',
  '储能电池':    '储能电池 新能源 电力系统 电池技术 光伏储能 锂电池 能源管理 充放电',
  'OKX交易':     'OKX 加密货币 比特币 以太坊 交易策略 数字资产 DeFi 区块链 行情分析',
  '其他学习':    '通用知识 学习资料 个人成长 技能提升 工具方法 其他领域 综合学习 知识积累',
};
const ORGANIZE_BATCH_MAX = 200; // 加速期：200（正常 100）

// ── v12 多模态知识库常量 ────────────────────────────────────────────────────────
const RECORD_TYPES = { KNOWLEDGE: 'knowledge', ENTITY: 'entity', RELATION: 'relation' };

const RELATION_TYPES = {
  SUPPORTS:     'supports',
  CONTRADICTS:  'contradicts',
  EXTENDS:      'extends',
  DEPENDS_ON:   'depends_on',
  USED_IN:      'used_in',
  EVOLVED_FROM: 'evolved_from',
  CROSS_DOMAIN: 'cross_domain',
};

const EMBED_SAFE_CHARS   = 6000;  // bge-m3 8192 token ≈ 6000中文字符安全上限
const CONFIDENCE_DEFAULT = 0.6;   // 新知识默认置信度

// 新增域（扩展 DOMAIN_DESCRIPTIONS，启动时合并）
const NEW_DOMAIN_DESCRIPTIONS = {
  '短视频生产': '短视频脚本 剪辑 钩子 开场 爆款 拍摄 BGM 字幕 封面 完播率',
  '自动化工具': '自动化 脚本 工作流 OpenClaw Claude API MCP 效率工具 代码',
  '人际沟通':   '微信聊天 话术 谈判 说服 客户沟通 回复模板 关系维护',
  '交易投资':   '加密货币 比特币 以太坊 交易信号 行情 止损 止盈 仓位 DeFi',
  '新闻热点':   '热搜 突发 今日热点 实时资讯 社会动态 行业新闻',
};
Object.assign(DOMAIN_DESCRIPTIONS, NEW_DOMAIN_DESCRIPTIONS);

const FRAGMENT_SIGNALS = [
  /第[一二三四五六七八九十\d]+[\/\-]\d+部分/,
  /part\s*\d+\s*of\s*\d+/i,
  /\(续\)|（续）|接上文|续上/,
  /待续|未完待续/,
  /\[\d+\/\d+\]/,
];

// ── 运行时状态 ─────────────────────────────────────────────────────────────────
const embedCache     = new Map();
let embedCacheHits   = 0;
let embedCacheMisses = 0;
const sessionTurns   = new Map();
let lastInjectKey    = '';
let lastInjectResult = undefined;
let lastBackupTime   = null;
let   lastInjectedIds        = [];             // ★ INJECT 注入的记忆 ID，供 feedback 定位
const distillWrittenHashes   = new Set();      // ★ distill 已写入的 hash，阻止 CAPTURE 二次捕获
const domainEmbeddingCache   = new Map();      // 域描述向量缓存（整理Agent）

// ── WriteQueue（并发安全，P1=CAPTURE/LEARN，P2=Agent）────────────────────────
class WriteQueue {
  constructor() { this._queue = []; this._running = false; }
  push(priority, fn) {
    return new Promise((resolve, reject) => {
      this._queue.push({ priority, fn, resolve, reject });
      this._queue.sort((a, b) => a.priority - b.priority);
      this._drain();
    });
  }
  async _drain() {
    if (this._running || !this._queue.length) return;
    this._running = true;
    while (this._queue.length) {
      const { fn, resolve, reject } = this._queue.shift();
      try { resolve(await fn()); } catch (e) { reject(e); }
    }
    this._running = false;
  }
}
const writeQueue = new WriteQueue();
const WRITE_PRIORITY = { CAPTURE: 1, LEARN: 1, AGENT: 2 };

// ── Agent 锁（防止同一 Agent 并发执行）───────────────────────────────────────
const agentLocks = new Map([
  ['organize', false], ['domain', false], ['associate', false],
  ['synthesize', false], ['meta', false], ['restructure', false],
  ['completion', false],
]);

async function runAgent(name, fn) {
  if (agentLocks.get(name)) return;
  agentLocks.set(name, true);
  try { await fn(); } finally { agentLocks.set(name, false); }
}

// ── HTTP/HTTPS 工具 ───────────────────────────────────────────────────────────
function httpReq(url, method = 'GET', body = null, extraHeaders = {}, timeoutMs = TIMEOUT_MS) {
  return new Promise((resolve) => {
    const u   = new URL(url);
    const mod = u.protocol === 'https:' ? https : http;
    const req = mod.request({
      hostname: u.hostname,
      port:     Number(u.port) || (u.protocol === 'https:' ? 443 : 80),
      path:     u.pathname + u.search,
      method,
      headers:  { 'Content-Type': 'application/json', ...extraHeaders },
      rejectUnauthorized: false,
    }, (res) => {
      let raw = '';
      res.on('data', c => { raw += c; });
      res.on('end', () => {
        try   { resolve({ ok: res.statusCode < 300, status: res.statusCode, body: JSON.parse(raw) }); }
        catch { resolve({ ok: res.statusCode < 300, status: res.statusCode, body: raw }); }
      });
    });
    req.setTimeout(timeoutMs, () => { req.destroy(); resolve({ ok: false, error: 'timeout' }); });
    req.on('error', e => resolve({ ok: false, error: e.message }));
    if (body !== null) req.write(JSON.stringify(body));
    req.end();
  });
}

// ── URL 网页抓取 ──────────────────────────────────────────────────────────────
function fetchUrlText(url, redirectsLeft = 3) {
  return new Promise((resolve) => {
    try {
      const u   = new URL(url);
      const mod = u.protocol === 'https:' ? https : http;
      const req = mod.request({
        hostname: u.hostname,
        port:     Number(u.port) || (u.protocol === 'https:' ? 443 : 80),
        path:     u.pathname + u.search,
        method:   'GET',
        headers:  { 'User-Agent': 'Mozilla/5.0 (compatible; atlas-memory/9.3)', 'Accept': 'text/html,text/plain,*/*' },
        rejectUnauthorized: false,
      }, (res) => {
        if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location && redirectsLeft > 0) {
          req.destroy();
          const next = res.headers.location.startsWith('http') ? res.headers.location : new URL(res.headers.location, url).href;
          resolve(fetchUrlText(next, redirectsLeft - 1));
          return;
        }
        let raw = '';
        res.setEncoding('utf8');
        res.on('data', c => { if (raw.length < 200_000) raw += c; });
        res.on('end', () => resolve({ ok: res.statusCode < 400, text: raw, contentType: res.headers['content-type'] ?? '' }));
      });
      req.setTimeout(FETCH_TIMEOUT_MS, () => { req.destroy(); resolve({ ok: false, error: 'timeout', text: '' }); });
      req.on('error', e => resolve({ ok: false, error: e.message, text: '' }));
      req.end();
    } catch (e) {
      resolve({ ok: false, error: e.message, text: '' });
    }
  });
}

function htmlToText(html) {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, '')
    .replace(/<style[\s\S]*?<\/style>/gi, '')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/\s{2,}/g, ' ').trim();
}

function isSearchTool(toolName) {
  const lower = toolName.toLowerCase();
  return SEARCH_TOOL_KEYWORDS.some(k => lower.includes(k));
}

function searchResultToText(result, query) {
  if (!result) return '';
  if (typeof result === 'string') return result;
  if (Array.isArray(result?.content)) {
    return result.content.filter(b => b?.type === 'text').map(b => b.text ?? '').join('\n');
  }
  if (Array.isArray(result?.results)) {
    return result.results.map(r =>
      [r.title, r.url, r.snippet ?? r.description ?? r.body ?? ''].filter(Boolean).join(' — ')
    ).join('\n');
  }
  try { return JSON.stringify(result); } catch { return ''; }
}

// ── ① 嵌入（LRU 缓存，bge-m3 支持 8192 token，中文约 6000 字安全上限）────────
async function embed(text) {
  const key = createHash('sha256').update(text.slice(0, 6000)).digest('hex').slice(0, 16);
  if (embedCache.has(key)) {
    embedCacheHits++;
    const v = embedCache.get(key);
    embedCache.delete(key);
    embedCache.set(key, v);
    return v;
  }
  embedCacheMisses++;
  const r = await httpReq(`${OLLAMA}/api/embeddings`, 'POST', {
    model: EMBED_MODEL, prompt: text.slice(0, 6000),
  });
  if (r.ok && Array.isArray(r.body?.embedding)) {
    if (embedCache.size >= EMBED_CACHE_SIZE) embedCache.delete(embedCache.keys().next().value);
    embedCache.set(key, r.body.embedding);
    return r.body.embedding;
  }
  return null;
}

// ── ★ omlx Qwen3.5-9B 推理（替代 Ollama 提取）───────────────────────────────
async function omlxGenerate(systemMsg, userMsg, maxTokens = 800, timeoutMs = EXTRACT_TIMEOUT_MS) {
  const r = await httpReq(
    `${OMLX}/v1/chat/completions`, 'POST',
    {
      model:       OMLX_MODEL,
      messages:    [
        { role: 'system', content: systemMsg },
        { role: 'user',   content: userMsg },
      ],
      temperature: 0.1,
      max_tokens:  maxTokens,
      stream:      false,
    },
    {},
    timeoutMs,
  );
  if (!r.ok) return null;
  return r.body?.choices?.[0]?.message?.content ?? null;
}
const AGENT_OMLX_TIMEOUT_MS = 90_000; // 后台Agent调用，允许更长等待

// ── ★ DeepSeek 云端推理（用于 distill 等复杂合成，token 有限制）─────────────
async function deepseekGenerate(systemMsg, userMsg, maxTokens = 400) {
  const key = DEEPSEEK_API_KEY || process.env.DEEPSEEK_API_KEY;
  if (!key) return null;
  const r = await httpReq(
    `${DEEPSEEK_URL}/v1/chat/completions`, 'POST',
    {
      model:       DEEPSEEK_MODEL,
      messages:    [
        { role: 'system', content: systemMsg },
        { role: 'user',   content: userMsg   },
      ],
      temperature: 0.3,
      max_tokens:  maxTokens,
      stream:      false,
    },
    { Authorization: `Bearer ${key}` },
    DEEPSEEK_TIMEOUT_MS,
  );
  if (!r.ok) return null;
  return r.body?.choices?.[0]?.message?.content ?? null;
}

// ── JSON 解析工具 ─────────────────────────────────────────────────────────────
function parseFactsJson(text) {
  if (!text) return [];
  const clean = text.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
  for (let i = 0; i < clean.length; i++) {
    if (clean[i] !== '[') continue;
    try {
      const parsed = JSON.parse(clean.slice(i));
      if (Array.isArray(parsed)) {
        return parsed.filter(f => typeof f?.content === 'string' && f.content.trim().length > 10);
      }
    } catch { /* try next */ }
  }
  return [];
}

function parseJsonObject(text) {
  if (!text) return null;
  // 去除 <think> 标签和 markdown 代码块包裹（DeepSeek 有时忽略"不要代码块"指令）
  let clean = text.replace(/<think>[\s\S]*?<\/think>/gi, '').trim();
  clean = clean.replace(/^```(?:json)?\s*/i, '').replace(/\s*```\s*$/g, '').trim();
  for (let i = 0; i < clean.length; i++) {
    if (clean[i] !== '{') continue;
    try {
      const parsed = JSON.parse(clean.slice(i));
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed;
    } catch { /* try next */ }
  }
  // 降级：JSON被截断时，尝试抢救已完整的nodes数组
  const nodesMatch = clean.match(/"nodes"\s*:\s*(\[[\s\S]*)/);
  if (nodesMatch) {
    const fragment = nodesMatch[1];
    // 尝试找最后一个完整的 } 闭合对象
    let depth = 0, lastCompleteEnd = -1;
    for (let i = 0; i < fragment.length; i++) {
      if (fragment[i] === '{') depth++;
      else if (fragment[i] === '}') { depth--; if (depth === 0) lastCompleteEnd = i; }
    }
    if (lastCompleteEnd > 0) {
      try {
        const repaired = JSON.parse(fragment.slice(0, lastCompleteEnd + 1) + ']');
        if (Array.isArray(repaired) && repaired.length > 0) {
          console.warn(`[parseJsonObject] JSON截断，抢救到 ${repaired.length} 个完整节点`);
          return { nodes: repaired, entities: [], relations: [] };
        }
      } catch { /* salvage failed */ }
    }
  }
  return null;
}

// ── ★ 事实提取（omlx + 质量过滤 + memory_type + 用户上下文）─────────────────
async function extractFacts(assistantText, userContext = '') {
  if (!assistantText || assistantText.length < getMinCaptureChars('unknown')) return [];
  const sys = '你是记忆提取专家。严格只输出有效JSON数组，不要任何解释或额外文字。';
  const contextLine = userContext
    ? `\n用户提问/上下文（辅助理解）：\n${userContext.slice(0, 400)}\n`
    : '';
  const user =
    `从以下对话中提取 0-5 条值得长期记忆的重要事实。${contextLine}
只提取跨会话有价值的内容（用户偏好、技术决策、项目约束、重要能力、关键结论）。
不提取：临时任务、本次对话特有内容、泛泛而谈的信息。
对每条打质量分(1-10)，只有≥${MIN_QUALITY_SCORE}分才值得保存。

格式（JSON数组）：
[{"content":"事实内容（简洁准确，20-200字）","category":"work|personal|project|system|learning","importance":"low|medium|high|critical","tags":["标签"],"quality":8,"memory_type":"preference|fact|skill|project|constraint|event"}]
无重要内容返回[]

助手回复：
${assistantText.slice(0, 3000)}

JSON数组：`;
  const out = await omlxGenerate(sys, user, 1000);
  if (!out) return [];
  const facts = parseFactsJson(out);
  return facts.filter(f => (f.quality ?? 10) >= MIN_QUALITY_SCORE);
}

// ── ★ 网页知识提取（omlx）────────────────────────────────────────────────────
async function extractWebFacts(text, query) {
  if (!text || text.length < 80) return [];
  const sys = '你是知识提取助手。严格只输出有效JSON数组，不要任何解释。';
  const user =
    `从以下网络内容中提取有价值的知识。
搜索查询: ${(query ?? '').slice(0, 100)}

只提取客观、持久有效的知识（技术事实、操作方法、重要定义、最佳实践）。
不提取：广告、导航、时效性内容。最多3条，质量<7的不提取。

格式：[{"content":"知识点","category":"learning","importance":"low|medium|high","tags":["主题"],"quality":7,"memory_type":"fact|skill|constraint"}]
无价值内容返回[]

内容：
${text.slice(0, 3000)}

JSON数组：`;
  const out = await omlxGenerate(sys, user, 600);
  if (!out) return [];
  const facts = parseFactsJson(out);
  return facts.filter(f => (f.quality ?? 10) >= MIN_QUALITY_SCORE);
}

// ── ★ 冲突检测与解决 ─────────────────────────────────────────────────────────
async function detectConflict(newContent, candidates) {
  if (!candidates.length) return null;
  const sys = '你是记忆冲突检测助手。严格只输出JSON对象，不要解释。';
  const list = candidates.map((c, i) => `${i + 1}. ${c.payload?.content ?? ''}`).join('\n');
  const user =
    `判断新记忆是否与旧记忆存在语义矛盾（而非仅仅是补充或相关）：

新记忆：${newContent}

旧记忆：
${list}

输出JSON：{"has_conflict":true/false,"conflict_index":0,"action":"keep_new|keep_old|merge","merged_content":""}
说明：conflict_index从1开始（0=无冲突）。has_conflict=false时其他字段忽略。
merge时merged_content填写合并后的完整内容。`;
  const out = await omlxGenerate(sys, user, 250);
  if (!out) return null;
  return parseJsonObject(out);
}

// ── v11 信息源类型检测 ────────────────────────────────────────────────────────
function detectSourceType(content = '', url = '', tags = []) {
  const text = (content + ' ' + tags.join(' ')).toLowerCase();

  const tradingPat = [/\bbtc\b/, /\beth\b/, /涨跌/, /开多/, /开空/, /入场/, /止损/, /止盈/, /ticker/, /仓位/, /行情/, /汇率/, /coin/];
  if (tradingPat.some(p => p.test(text))) return 'trading';

  const newsPat = [/今日热点/, /突发/, /热搜/, /刚刚/, /最新消息/, /breaking/, /据报道/];
  if (newsPat.some(p => p.test(text))) return 'news';

  const platformPat = [/算法/, /完播率/, /推流/, /流量池/, /\bctr\b/, /违禁词/, /审核/, /限流/, /平台规则/];
  if (platformPat.some(p => p.test(text))) return 'platform_rule';

  const processPat = [/步骤\d/, /第[一二三四五六七八九十\d]+步/, /操作指南/, /工作流/, /\bsop\b/];
  if (processPat.some(p => p.test(text))) return 'process';

  const chatPat = [/回复模板/, /话术/, /怎么回/, /聊天技巧/, /沟通模板/];
  if (chatPat.some(p => p.test(text))) return 'chat';

  const socialPat = [/脚本/, /钩子/, /爆款/, /标题公式/, /开头/, /文案结构/];
  if (socialPat.some(p => p.test(text))) return 'social';

  if (url) {
    // trading 优先（okx.com 不能被 x.com 误匹配，所以 trading 在前）
    if (/finance|eastmoney|xueqiu|binance|okx\.com/.test(url)) return 'trading';
    if (/(?:\/\/)x\.com|twitter\.com|weibo|douyin|xiaohongshu|bilibili/.test(url)) return 'social';
    if (/news|xinhua|163\.com\/dy/.test(url)) return 'news';
  }

  if (content.includes('> ') && content.includes('####')) return 'course';

  return 'unknown';
}

// ── v11 知识颗粒度方向检测 ────────────────────────────────────────────────────
function detectKnowledgePurpose(content = '', tags = [], sourceType = 'unknown') {
  if (sourceType === 'process') return KNOWLEDGE_PURPOSE.PROCESS;
  if (/步骤\d|第[一二三四五]步|sop|操作流程/.test(content.toLowerCase())) return KNOWLEDGE_PURPOSE.PROCESS;
  if (['social', 'chat', 'platform_rule'].includes(sourceType)) return KNOWLEDGE_PURPOSE.PRODUCTION;
  if (/模板|公式|钩子句|填空|示例文案|脚本框架/.test(content)) return KNOWLEDGE_PURPOSE.PRODUCTION;
  return KNOWLEDGE_PURPOSE.UNDERSTANDING;
}

// ── v11 TTL 过期时间计算 ───────────────────────────────────────────────────────
function calcTTLExpiry(sourceType) {
  const ttlSeconds = TTL_MAP[sourceType];
  if (ttlSeconds === null || ttlSeconds === undefined) return null;
  return new Date(Date.now() + ttlSeconds * 1000).toISOString();
}

// ── v11 按 source_type 获取最小捕获字符数 ────────────────────────────────────
function getMinCaptureChars(sourceType) {
  return MIN_CAPTURE_CHARS_MAP[sourceType] ?? MIN_CAPTURE_CHARS_MAP.unknown;
}

// ── 时间衰减评分（v11：意图感知，修复精度问题）────────────────────────────────
function applyTimeDecay(hits, intent = 'relevant') {
  if (intent === 'latest') {
    // 最新意图：按创建时间倒序，不用语义分数
    return [...hits].sort((a, b) => {
      const ta = a.payload?.created_at ? new Date(a.payload.created_at).getTime() : 0;
      const tb = b.payload?.created_at ? new Date(b.payload.created_at).getTime() : 0;
      return tb - ta;
    }).map(h => ({ ...h, effectiveScore: h.score }));
  }
  // 相关意图（默认）：不做时间惩罚，freshness_score 过滤已在 qdrantSearch 处理
  return hits.map(h => ({ ...h, effectiveScore: h.score }))
             .sort((a, b) => b.effectiveScore - a.effectiveScore);
}

// ── v11 衰减速率推断（source_type 优先）──────────────────────────────────────
function inferDecayRate(domain, tags = [], sourceType = null) {
  if (sourceType === 'trading' || sourceType === 'news') return 'fast';
  if (sourceType === 'social' || sourceType === 'chat' || sourceType === 'platform_rule') return 'medium';
  if (sourceType === 'course' || sourceType === 'process') return 'slow';

  const fastDomains = ['OKX交易'];
  const fastKeywords = ['价格', '行情', '市场', '汇率', 'price', 'market', '报价'];
  if (fastDomains.includes(domain)) return 'fast';
  if (tags.some(t => fastKeywords.some(k => t.includes(k)))) return 'fast';

  const slowDomains = ['战略', '情感学'];
  const slowKeywords = ['原则', '规律', '方法论', '底层逻辑', '核心', '战略', '框架', '模型'];
  if (slowDomains.includes(domain)) return 'slow';
  if (tags.some(t => slowKeywords.some(k => t.includes(k)))) return 'slow';

  return 'medium';
}

// ── 访问计数 + 自动升级 importance ────────────────────────────────────────────
// confidence 随 hit_count 增长（指数逼近，永不到1.0）
// hit=0→0.60, hit=1→0.66, hit=5→0.82, hit=10→0.92, hit=20→0.98
function calcConfidence(hitCount) {
  return parseFloat(Math.min(0.99, 1 - (1 - CONFIDENCE_DEFAULT) * Math.pow(0.85, hitCount)).toFixed(2));
}

async function trackAccess(hits) {
  for (const h of hits) {
    const hitCount  = (h.payload?.hit_count ?? 0) + 1;
    const imp       = h.payload?.importance ?? 'medium';
    const impIdx    = IMPORTANCE_LEVELS.indexOf(imp);
    const threshold = HIT_UPGRADE[imp];
    const newImp    = (threshold && hitCount >= threshold && impIdx < IMPORTANCE_LEVELS.length - 1)
      ? IMPORTANCE_LEVELS[impIdx + 1] : imp;
    const newConf   = calcConfidence(hitCount);
    const update    = {
      hit_count:        hitCount,
      last_accessed_at: new Date().toISOString(),
      confidence:       newConf,
    };
    if (newImp !== imp) {
      update.importance = newImp;
      appendEvolutionLog('UPGRADE', `"${(h.payload?.content ?? '').slice(0, 50)}" ${imp}→${newImp} confidence=${newConf}（访问${hitCount}次）`).catch(() => {});
    }
    await httpReq(
      `${QDRANT}/collections/${COLLECTION}/points/payload`, 'POST',
      { payload: update, points: [h.id] },
    );
  }
}

// ── 文本分块 ──────────────────────────────────────────────────────────────────
function chunkText(text, size = CHUNK_SIZE, overlap = CHUNK_OVERLAP) {
  if (text.length <= size) return [text];
  const chunks = [];
  let i = 0;
  while (i < text.length) {
    chunks.push(text.slice(i, i + size));
    if (i + size >= text.length) break;
    i += size - overlap;
  }
  return chunks;
}

// ── ★ 从 messages 提取对话上下文（含用户消息）───────────────────────────────
function extractConversationContext(messages) {
  if (!Array.isArray(messages)) return { assistantText: '', userContext: '' };
  const recent = messages.slice(-12);
  const getText = (msg) => {
    const c = msg.content;
    if (typeof c === 'string') return c;
    if (Array.isArray(c)) return c.filter(b => b?.type === 'text').map(b => b.text ?? '').join(' ');
    return '';
  };
  const userTexts      = recent.filter(m => m?.role === 'user').map(getText).filter(t => t.trim());
  const assistantTexts = recent.filter(m => m?.role === 'assistant').map(getText).filter(t => t.trim());
  return {
    assistantText: assistantTexts.slice(-2).join('\n\n').trim(),
    userContext:   userTexts.slice(-3).join('\n').trim(),
  };
}

// ── Qdrant 操作 ───────────────────────────────────────────────────────────────
async function ensureCollection() {
  const check = await httpReq(`${QDRANT}/collections/${COLLECTION}`);
  if (check.ok) return true;
  const create = await httpReq(`${QDRANT}/collections/${COLLECTION}`, 'PUT', {
    vectors:           { size: VECTOR_DIM, distance: 'Cosine' },
    on_disk_payload:   true,
    optimizers_config: { memmap_threshold: 20000 },
  });
  if (!create.ok) return false;
  // v11 新字段索引（方便过滤，异步创建，失败不影响主流程）
  const indexDefs = [
    { field_name: 'source_type',       field_schema: 'keyword' },
    { field_name: 'knowledge_purpose', field_schema: 'keyword' },
    { field_name: 'platform',          field_schema: 'keyword' },
    { field_name: 'expires_at',        field_schema: 'datetime' },
    { field_name: 'domain',            field_schema: 'keyword' },
    { field_name: 'level',             field_schema: 'integer' },
    { field_name: 'status',            field_schema: 'keyword' },
  ];
  await Promise.allSettled(indexDefs.map(def =>
    httpReq(`${QDRANT}/collections/${COLLECTION}/index`, 'PUT', def)
  ));
  return true;
}

function stableId(text) {
  return parseInt(createHash('sha256').update(text).digest('hex').slice(0, 15), 16);
}

// ── v12 实体注册 ──────────────────────────────────────────────────────────────
async function upsertEntity({ canonical_name, aliases = [], domains = [], definition = '', related_entity_names = [] }) {
  try {
    const scrollRes = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', {
      filter: { must: [
        { key: 'record_type', match: { value: RECORD_TYPES.ENTITY } },
        { key: 'canonical_name', match: { value: canonical_name } },
      ]},
      limit: 1, with_payload: true, with_vector: false,
    });
    const existing = scrollRes?.body?.result?.points?.[0];
    if (existing) {
      await httpReq(`${QDRANT}/collections/${COLLECTION}/points/payload`, 'POST', {
        payload: {
          aliases:               [...new Set([...(existing.payload.aliases ?? []), ...aliases])],
          domains:               [...new Set([...(existing.payload.domains ?? []), ...domains])],
          related_entity_names:  [...new Set([...(existing.payload.related_entity_names ?? []), ...related_entity_names])],
          updated_at:            new Date().toISOString(),
        },
        points: [existing.id],
      });
      return { ok: true, entity_id: existing.id, merged: true };
    }
    const entityPointId = stableId(canonical_name + '_entity');
    const vector = await embed((canonical_name + ' ' + definition).slice(0, EMBED_SAFE_CHARS));
    if (!vector) return { ok: false, error: 'embed failed' };
    const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points?wait=true`, 'PUT', {
      points: [{ id: entityPointId, vector, payload: {
        record_type: RECORD_TYPES.ENTITY, canonical_name, aliases, domains, definition,
        related_entity_names, knowledge_node_ids: [], confidence: CONFIDENCE_DEFAULT,
        created_at: new Date().toISOString(), updated_at: new Date().toISOString(),
      }}],
    });
    return { ok: r.ok, entity_id: entityPointId };
  } catch (err) { return { ok: false, error: err?.message ?? String(err) }; }
}

async function upsertRelation({ source_id, target_id, relation_type, strength = 0.7, context = '' }) {
  try {
    const relationPointId = stableId(String(source_id) + '_' + relation_type + '_' + String(target_id));
    const embedText = (context || `${source_id} ${relation_type} ${target_id}`).slice(0, EMBED_SAFE_CHARS);
    const vector = await embed(embedText);
    if (!vector) return { ok: false, error: 'embed failed' };
    const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points?wait=true`, 'PUT', {
      points: [{ id: relationPointId, vector, payload: {
        record_type: RECORD_TYPES.RELATION, source_id, target_id,
        relation_type, strength, context, created_at: new Date().toISOString(),
      }}],
    });
    return { ok: r.ok, relation_id: relationPointId };
  } catch (err) { return { ok: false, error: err?.message ?? String(err) }; }
}

async function upsert(vector, payload) {
  const id = stableId(payload.content + (payload.created_at ?? ''));
  const r  = await httpReq(`${QDRANT}/collections/${COLLECTION}/points?wait=true`, 'PUT', {
    points: [{ id, vector, payload }],
  });
  return { ok: r.ok, id, error: r.error };
}

async function qdrantSearch(vector, {
  limit = 5, category, domain, minScore = SCORE_MIN, source_type, platform,
  intent = 'relevant', min_confidence = 0, expand_entities = false, record_type = null,
} = {}) {
  const body = { vector, limit, with_payload: true, score_threshold: minScore };
  const now = new Date().toISOString();
  const mustNot = [
    { key: 'status', match: { value: 'superseded' } },
    { key: 'status', match: { value: 'archived' } },
    { key: 'expires_at', range: { lt: now } },
  ];
  const must = [];
  if (category && category !== 'any') must.push({ key: 'category', match: { value: category } });
  if (domain && domain !== 'any') must.push({ key: 'domain', match: { value: domain } });
  if (source_type) must.push({ key: 'source_type', match: { value: source_type } });
  if (platform) must.push({ key: 'platform', match: { text: platform } });
  if (record_type) {
    must.push({ key: 'record_type', match: { value: record_type } });
  } else {
    mustNot.push({ key: 'record_type', match: { value: RECORD_TYPES.ENTITY } });
    mustNot.push({ key: 'record_type', match: { value: RECORD_TYPES.RELATION } });
  }
  if (min_confidence > 0) must.push({ key: 'confidence', range: { gte: min_confidence } });
  body.filter = must.length ? { must, must_not: mustNot } : { must_not: mustNot };
  const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/search`, 'POST', body);
  if (!r.ok) return [];
  const baseHits = (r.body?.result ?? []).filter(h =>
    (h.payload?.feedback_score  ?? 1.0) >= FEEDBACK_FILTER_MIN &&
    (h.payload?.freshness_score ?? 1.0) >= FRESHNESS_INJECT_MIN
  );
  if (!expand_entities || baseHits.length === 0) return baseHits;
  try {
    const entityIds = [...new Set(baseHits.flatMap(h => h.payload?.entity_ids ?? []).filter(Boolean))];
    if (!entityIds.length) return baseHits;
    const scrollR = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', {
      limit: limit * 2, with_payload: true,
      filter: { must: [{ key: 'entity_ids', match: { any: entityIds } }], must_not: mustNot },
    });
    if (!scrollR.ok) return baseHits;
    const expanded = (scrollR.body?.result?.points ?? []).filter(h =>
      (h.payload?.feedback_score  ?? 1.0) >= FEEDBACK_FILTER_MIN &&
      (h.payload?.freshness_score ?? 1.0) >= FRESHNESS_INJECT_MIN
    );
    const seen = new Set(baseHits.map(h => h.id));
    const merged = [...baseHits];
    for (const h of expanded) { if (!seen.has(h.id)) { seen.add(h.id); merged.push({ ...h, score: h.score ?? 0 }); } }
    return merged.sort((a, b) => (b.score ?? 0) - (a.score ?? 0)).slice(0, limit);
  } catch { return baseHits; }
}

async function qdrantDelete(ids) {
  if (!ids.length) return { ok: true, deleted: 0 };
  const r = await httpReq(
    `${QDRANT}/collections/${COLLECTION}/points/delete?wait=true`, 'POST', { points: ids },
  );
  return { ok: r.ok, deleted: ids.length };
}

async function qdrantPatchPayload(id, patch) {
  return httpReq(
    `${QDRANT}/collections/${COLLECTION}/points/payload`, 'POST',
    { payload: patch, points: [id] },
  );
}

function fmtHits(hits) {
  return hits.map(h => ({
    id:               h.id,               // ★ 供 atlas_feedback 定位使用
    content:          h.payload?.content,
    category:         h.payload?.category,
    importance:       h.payload?.importance,
    memory_type:      h.payload?.memory_type ?? 'fact',
    tags:             h.payload?.tags ?? [],
    source:           h.payload?.source ?? 'unknown',
    created_at:       h.payload?.created_at,
    hit_count:        h.payload?.hit_count ?? 0,
    last_accessed_at: h.payload?.last_accessed_at,
    score:            Math.round((h.effectiveScore ?? h.score) * 1000) / 1000,
  }));
}

// v10 INJECT：层级优先搜索（L3→L2→L1→L0）
// 两路搜索：Pass1 全局 threshold=0.65；Pass2 专门针对 L2/L3 threshold=0.45
async function qdrantSearchForInject(vector) {
  const notSuperseded = { must_not: [{ key: 'status', match: { value: 'superseded' } }] };

  // Pass 1: flat search across all levels
  const r1 = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/search`, 'POST', {
    vector, limit: INJECT_LIMIT * 3, with_payload: true,
    score_threshold: SCORE_MIN, filter: notSuperseded,
  });
  const raw1 = r1.ok ? (r1.body?.result ?? []) : [];

  // Pass 2: dedicated L2/L3 search with lower threshold to surface synthesized knowledge
  const r2 = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/search`, 'POST', {
    vector, limit: INJECT_LIMIT, with_payload: true,
    score_threshold: 0.45,
    filter: {
      must: [{ key: 'level', match: { any: [LEVEL_INSIGHT, LEVEL_WISDOM, LEVEL_META] } }],
      must_not: notSuperseded.must_not,
    },
  });
  const raw2 = r2.ok ? (r2.body?.result ?? []) : [];

  // Merge, deduplicate by ID
  const seen = new Set();
  const merged = [];
  for (const h of [...raw1, ...raw2]) {
    const key = String(h.id);
    if (!seen.has(key)) { seen.add(key); merged.push(h); }
  }

  // 过滤：负反馈 + 低新鲜度
  const filtered = merged.filter(h =>
    (h.payload?.feedback_score  ?? 1.0) >= FEEDBACK_FILTER_MIN &&
    (h.payload?.freshness_score ?? 1.0) >= FRESHNESS_INJECT_MIN
  );
  // 排序：level DESC → score DESC → importance DESC
  filtered.sort((a, b) => {
    const la = a.payload?.level ?? 0;
    const lb = b.payload?.level ?? 0;
    if (lb !== la) return lb - la;
    const sa = a.effectiveScore ?? a.score ?? 0;
    const sb = b.effectiveScore ?? b.score ?? 0;
    if (Math.abs(sb - sa) > 0.01) return sb - sa;
    const ia = IMPORTANCE_ORDER[a.payload?.importance] ?? 2;
    const ib = IMPORTANCE_ORDER[b.payload?.importance] ?? 2;
    return ib - ia;
  });
  return filtered.slice(0, INJECT_LIMIT);
}

// 尝试读 Obsidian 源文件，带超时保护
async function tryReadObsidianFile(obsidianPath, budgetMs = 500) {
  if (!OBSIDIAN_VAULT || !obsidianPath) return null;
  try {
    const fullPath = join(OBSIDIAN_VAULT, obsidianPath);
    const text = await Promise.race([
      readFile(fullPath, 'utf8'),
      new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), budgetMs)),
    ]);
    // Strip frontmatter
    return text.replace(/^---[\s\S]*?---\n/, '').trim().slice(0, 600);
  } catch { return null; }
}

// v10 formatInjectContext：层级分区 + 源文件内容
function formatInjectContext(hits, fileContents = {}) {
  if (!hits.length) return '';

  const byLevel = { 4: [], 3: [], 2: [], 1: [], 0: [] };
  for (const h of hits) {
    const lvl = h.payload?.level ?? 0;
    byLevel[Math.min(lvl, 4)].push(h);
  }

  const levelLabel = { 4: '【L4 元规律】', 3: '【L3 智识框架】', 2: '【L2 跨域洞见】', 1: '【L1 知识】', 0: '【L0 原料】' };
  const sections = [];

  for (const lvl of [4, 3, 2, 1, 0]) {
    const group = byLevel[lvl];
    if (!group.length) continue;
    sections.push(levelLabel[lvl]);
    for (const h of group) {
      const domain = h.payload?.domain ? `[${h.payload.domain}]` : '';
      const topic  = h.payload?.topic  ? ` · ${h.payload.topic}` : '';
      const full   = fileContents[h.id];
      const body   = full ?? (h.payload?.content ?? '');
      sections.push(`${domain}${topic}\n${body}`);
    }
  }

  return `<atlas_memory>\n${sections.join('\n\n')}\n</atlas_memory>`;
}

// ── ★ 知识提炼内部存储 ────────────────────────────────────────────────────────
async function _storeDistilled(tag, content, basis, sourceDomain = null) {
  const vector = await embed(content);
  if (!vector) return null;
  await ensureCollection();
  const now  = new Date().toISOString();
  const tags = [tag, DISTILL_TAG];
  const payload = {
    content,
    category:           'work',
    importance:         'high',
    tags,
    memory_type:        'skill',
    created_at:         now,
    source:             'distill',
    session_key:        'system',
    hit_count:          0,
    last_accessed_at:   null,
    status:             'active',
    feedback_score:     1.0,
    distill_basis:      basis,
    level:              LEVEL_WISDOM,
    domain:             sourceDomain,   // 继承源记录的主域，不留 null
    topic:              tag,
    freshness_score:    1.0,
    decay_rate:         inferDecayRate(null, tags),
    last_verified:      now,
    source_ids:         [],
    associated_ids:     [],
    derived_to_id:      null,
    obsidian_path:      null,
    acquisition_source: 'distill',
  };
  const h = createHash('sha256').update(content.slice(0, 200)).digest('hex').slice(0, 16);
  distillWrittenHashes.add(h);
  const r = await upsert(vector, payload);
  return r.ok ? { ok: true, id: r.id, content, basis } : null;
}

// ── ★ 知识提炼主流程（DeepSeek 优先，omlx 备用）─────────────────────────────
async function distillTagMemories(tag, logger, force = false) {
  // 1. 拉取该标签下的非superseded、非distilled记忆
  let offset = null;
  const tagPoints = [];
  do {
    const body = {
      limit: 50, with_payload: true, with_vector: false,
      filter: {
        must: [{ key: 'tags', match: { value: tag } }],
        must_not: [
          { key: 'status', match: { value: 'superseded' } },
          { key: 'tags',   match: { value: DISTILL_TAG  } },
        ],
      },
    };
    if (offset != null) body.offset = offset;
    const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', body);
    if (!r.ok) break;
    tagPoints.push(...(r.body?.result?.points ?? []));
    offset = r.body?.result?.next_page_offset ?? null;
  } while (offset != null && tagPoints.length < 100);

  if (tagPoints.length < DISTILL_MIN_COUNT) {
    return { ok: false, skipped: true, reason: `"${tag}" 下只有 ${tagPoints.length} 条记忆，需要 ≥${DISTILL_MIN_COUNT} 条` };
  }

  // 2. 检查是否已有通则（force=true 时跳过检查）
  if (!force) {
    const checkR = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', {
      limit: 5, with_payload: false, with_vector: false,
      filter: {
        must: [
          { key: 'tags', match: { value: tag        } },
          { key: 'tags', match: { value: DISTILL_TAG } },
        ],
        must_not: [{ key: 'status', match: { value: 'superseded' } }],
      },
    });
    if (checkR.ok && (checkR.body?.result?.points ?? []).length > 0) {
      return { ok: false, skipped: true, reason: `"${tag}" 已有通则，使用 force:true 强制重新提炼` };
    }
  }

  // 3. 构建提炼提示（token 控制：最多 10 条，截断至 1800 字符）
  const topMems = tagPoints
    .sort((a, b) => (b.payload?.hit_count ?? 0) - (a.payload?.hit_count ?? 0))
    .slice(0, 10)
    .map((p, i) => `${i + 1}. [${p.payload?.importance ?? 'medium'}] ${(p.payload?.content ?? '').slice(0, 150)}`)
    .join('\n');

  const sys  = '你是知识提炼专家。从多条经验中提炼出一条简洁、可直接应用的"通则"。只输出通则内容，不超过150字，不要编号和解释。';
  const user = `标签：${tag}\n\n原始经验：\n${topMems.slice(0, 1800)}\n\n提炼通则：`;

  // 4. 调用 DeepSeek（优先）或 omlx（备用）
  let principle = null;
  if (DEEPSEEK_API_KEY) {
    principle = await deepseekGenerate(sys, user, 300);
  }
  if (!principle?.trim()) {
    logger?.warn?.('[atlas-memory] distill: DeepSeek 不可用，回退 omlx');
    principle = await omlxGenerate(sys, user, 200);
  }
  if (!principle?.trim()) return null;

  // 提取源记录的主域（频率最高的那个），传递给 _storeDistilled
  const domains = tagPoints.map(p => p.payload?.domain).filter(Boolean);
  const dominantDomain = domains.length
    ? domains.sort((a, b) => domains.filter(d => d === b).length - domains.filter(d => d === a).length)[0]
    : null;

  return await _storeDistilled(tag, principle.trim(), tagPoints.length, dominantDomain);
}

// ── ★ 带冲突检测的存储 ────────────────────────────────────────────────────────
async function storeWithConflict({ content, category = 'work', importance = 'medium', tags = [], memory_type = 'fact', source = 'manual', sessionKey, doConflictCheck = false }) {
  const vector = await embed(content);
  if (!vector) return { ok: false, error: 'Ollama embed 不可用' };

  // 精确去重
  const exactDup = await qdrantSearch(vector, { limit: 1, minScore: SCORE_DEDUP });
  if (exactDup.length) return { ok: true, deduplicated: true, similar: exactDup[0].payload?.content?.slice(0, 80) };

  let supersededId = null;  // ★ 版本化：记录被替换的旧版本 ID
  // ★ 冲突检测（仅 medium 以上重要性，避免对低价值内容浪费 omlx）
  if (doConflictCheck && IMPORTANCE_LEVELS.indexOf(importance) >= 1) {
    const candidates = await qdrantSearch(vector, { limit: 3, minScore: SCORE_CONFLICT_MIN });
    const conflicts  = candidates.filter(c => c.score < SCORE_DEDUP);
    if (conflicts.length > 0) {
      const res = await detectConflict(content, conflicts);
      if (res?.has_conflict) {
        const cidx = (res.conflict_index ?? 1) - 1;
        const conflictId = conflicts[Math.max(0, Math.min(cidx, conflicts.length - 1))]?.id;
        if (res.action === 'keep_old') {
          return { ok: true, skipped: true, reason: 'conflict_keep_old' };
        } else if (res.action === 'keep_new' && conflictId) {
          supersededId = conflictId;  // ★ 版本化：不删除，标记为 superseded
          await qdrantPatchPayload(conflictId, { status: 'superseded', superseded_at: new Date().toISOString() });
        } else if (res.action === 'merge' && res.merged_content?.trim() && conflictId) {
          supersededId = conflictId;  // ★ 版本化：合并时也保留旧版本
          await qdrantPatchPayload(conflictId, { status: 'superseded', superseded_at: new Date().toISOString() });
          content = res.merged_content.trim();
        }
      }
    }
  }

  await ensureCollection();
  const now     = new Date().toISOString();
  const tagList = Array.isArray(tags) ? tags : [];
  const payload = {
    content:            content.trim(),
    category,
    importance,
    tags:               tagList,
    memory_type,
    created_at:         now,
    source,
    session_key:        sessionKey ?? 'unknown',
    hit_count:          0,
    last_accessed_at:   null,
    status:             'active',
    feedback_score:     1.0,
    level:              LEVEL_KNOWLEDGE,
    domain:             null,
    topic:              tagList[0] ?? category ?? 'general',
    freshness_score:    1.0,
    decay_rate:         inferDecayRate(null, tagList),
    last_verified:      now,
    source_ids:         [],
    associated_ids:     [],
    derived_to_id:      null,
    obsidian_path:      null,
    acquisition_source: source,
  };
  const result = await upsert(vector, payload);
  if (result.ok && supersededId) {
    await qdrantPatchPayload(supersededId, { superseded_by: result.id }).catch(() => {});
  }
  return result.ok ? { ok: true, id: result.id } : { ok: false, error: result.error || 'Qdrant 写入失败' };
}

// ★ 向后兼容的 storeMemory（手动工具调用，启用冲突检测）
async function storeMemory(params) {
  return storeWithConflict({ ...params, doConflictCheck: true });
}

// ── ⑥ 批量存储（并行 embed → 逐条 intakeToL0，统一 L0 入口）──────────────────
async function batchStoreMemories(facts, source, sessionKey, doConflictCheck = false) {
  if (!facts.length) return { stored: 0, deduplicated: 0, skipped: 0 };
  await ensureCollection();

  // 并行 embed（保持性能）
  const embedded = await Promise.all(
    facts.map(async f => {
      const vector = await embed(f.content);
      return vector ? { f, vector } : null;
    })
  );
  const valid = embedded.filter(Boolean);
  let stored = 0, deduplicated = 0, skipped = 0;

  for (const { f, vector } of valid) {
    // 跳过 distill 刚写入的内容（防 CAPTURE 二次捕获）
    const ch = createHash('sha256').update(f.content.trim().slice(0, 200)).digest('hex').slice(0, 16);
    if (distillWrittenHashes.has(ch)) { deduplicated++; continue; }
    // 精确去重
    const dup = await qdrantSearch(vector, { limit: 1, minScore: SCORE_DEDUP });
    if (dup.length) { deduplicated++; continue; }

    // 冲突检测（medium+ 重要性）
    let content      = f.content.trim();
    const importance = f.importance ?? 'medium';
    let supersededId = null;
    if (doConflictCheck && IMPORTANCE_LEVELS.indexOf(importance) >= 1) {
      const candidates = await qdrantSearch(vector, { limit: 3, minScore: SCORE_CONFLICT_MIN });
      const conflicts  = candidates.filter(c => c.score < SCORE_DEDUP);
      if (conflicts.length > 0) {
        const res = await detectConflict(content, conflicts);
        if (res?.has_conflict) {
          const cidx       = (res.conflict_index ?? 1) - 1;
          const conflictId = conflicts[Math.max(0, Math.min(cidx, conflicts.length - 1))]?.id;
          if (res.action === 'keep_old') { skipped++; continue; }
          if ((res.action === 'keep_new' || res.action === 'merge') && conflictId) {
            supersededId = conflictId;
            await qdrantPatchPayload(conflictId, { status: 'superseded', superseded_at: new Date().toISOString() });
          }
          if (res.action === 'merge' && res.merged_content?.trim()) content = res.merged_content.trim();
        }
      }
    }

    // ★ v10 统一入口：Obsidian L0 + Qdrant（含 level/domain/freshness_score/decay_rate）
    const result = await intakeToL0({
      content,
      domain:         null,            // Phase 4 域检测Agent 自动填充
      topic:          f.tags?.[0] ?? f.category ?? 'general',
      source,
      tags:           Array.isArray(f.tags) ? f.tags : [],
      category:       f.category   ?? 'work',
      importance,
      memory_type:    f.memory_type ?? 'fact',
      knowledge_type: f.knowledge_type ?? 'capture',
      sessionKey,
    });

    if (result?.ok) {
      stored++;
      if (supersededId && result.id) {
        await qdrantPatchPayload(supersededId, { superseded_by: result.id }).catch(() => {});
      }
    }
  }

  if (stored > 0) {
    appendEvolutionLog('CAPTURE', `+${stored} 条记忆（${source}，去重${deduplicated}，冲突跳过${skipped}）`).catch(() => {});
  }
  return { stored, deduplicated, skipped };
}

// ── 后台进化：去重 + ★ 过期清理 ──────────────────────────────────────────────
async function runEvolution(logger) {
  logger?.info?.('[atlas-memory] 开始记忆进化（去重 + 过期清理）...');
  let offset = null;
  const allIds = [];
  do {
    const body = { limit: 250, with_payload: false, with_vector: false };
    if (offset != null) body.offset = offset;
    const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', body);
    if (!r.ok) break;
    allIds.push(...(r.body?.result?.points ?? []).map(p => p.id));
    offset = r.body?.result?.next_page_offset ?? null;
  } while (offset != null);

  if (allIds.length < 2) {
    logger?.info?.(`[atlas-memory] 进化完成：总数 ${allIds.length}，无需处理`);
    return { total: allIds.length, removed: 0, pruned: 0 };
  }

  const pointsWithVecs = [];
  for (let i = 0; i < allIds.length; i += 50) {
    const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points`, 'POST', {
      ids: allIds.slice(i, i + 50), with_vector: true, with_payload: true,
    });
    if (r.ok) pointsWithVecs.push(...(r.body?.result ?? []));
  }

  const toDelete = new Set();
  const now      = Date.now();

  // ★ 过期清理：hit_count=0 + age>90天 + importance='low'（跳过历史版本）
  for (const pt of pointsWithVecs) {
    if (pt.payload?.status === 'superseded') continue;  // ★ 版本历史永不过期清理
    const hitCount  = pt.payload?.hit_count ?? 0;
    const imp       = pt.payload?.importance ?? 'medium';
    const created   = pt.payload?.created_at ? new Date(pt.payload.created_at).getTime() : now;
    const ageDays   = (now - created) / 86_400_000;
    if (hitCount === 0 && imp === 'low' && ageDays > STALE_AGE_DAYS) {
      toDelete.add(pt.id);
    }
  }

  // 相似度去重（跳过历史版本）
  for (const pt of pointsWithVecs) {
    if (toDelete.has(pt.id) || !pt.vector) continue;
    if (pt.payload?.status === 'superseded') continue;  // ★ 不对历史版本做去重
    const similar = await qdrantSearch(pt.vector, { limit: 5, minScore: SCORE_DEDUP });
    for (const hit of similar) {
      if (hit.id === pt.id || toDelete.has(hit.id)) continue;
      const ptImp  = IMPORTANCE_LEVELS.indexOf(pt.payload?.importance  ?? 'medium');
      const hitImp = IMPORTANCE_LEVELS.indexOf(hit.payload?.importance ?? 'medium');
      toDelete.add(hitImp > ptImp ? pt.id : hit.id);
    }
  }

  if (toDelete.size > 0) {
    await qdrantDelete([...toDelete]);
    appendEvolutionLog('PRUNE', `清理 ${toDelete.size} 条记忆（过期/重复），剩余 ${allIds.length - toDelete.size} 条`).catch(() => {});
  }
  // ★ 自动提炼：统计标签分布，对积累 ≥ DISTILL_MIN_COUNT 条的标签自动生成通则
  const tagCount = new Map();
  for (const pt of pointsWithVecs) {
    if (pt.payload?.status === 'superseded') continue;
    if ((pt.payload?.tags ?? []).includes(DISTILL_TAG)) continue;
    for (const tag of (pt.payload?.tags ?? [])) {
      if (tag !== DISTILL_TAG) tagCount.set(tag, (tagCount.get(tag) ?? 0) + 1);
    }
  }
  const distillCandidates = [];
  for (const [tag, count] of tagCount) {
    if (count >= DISTILL_MIN_COUNT) {
      const hasDistilled = pointsWithVecs.some(pt =>
        !pt.payload?.status?.includes('superseded') &&
        (pt.payload?.tags ?? []).includes(DISTILL_TAG) &&
        (pt.payload?.tags ?? []).includes(tag)
      );
      if (!hasDistilled) distillCandidates.push({ tag, count });
    }
  }
  let distilled = 0;
  for (const { tag } of distillCandidates.sort((a, b) => b.count - a.count).slice(0, 3)) {
    const r = await distillTagMemories(tag, logger).catch(() => null);
    if (r?.ok) {
      distilled++;
      appendEvolutionLog('DISTILL', `自动提炼"${tag}"：${r.basis}条 → 通则（id:${String(r.id)?.slice(0, 8)}）`).catch(() => {});
    }
  }
  logger?.info?.(`[atlas-memory] 进化完成：总数 ${allIds.length}，删除 ${toDelete.size} 条，自动提炼 ${distilled} 条通则`);
  return { total: allIds.length, removed: toDelete.size, distilled };
}

// ── ⑦ 备份 ───────────────────────────────────────────────────────────────────
async function backupCollection(logger, customPath) {
  let offset = null;
  const points = [];
  do {
    const body = { limit: 250, with_payload: true, with_vector: true };
    if (offset != null) body.offset = offset;
    const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', body);
    if (!r.ok) break;
    points.push(...(r.body?.result?.points ?? []));
    offset = r.body?.result?.next_page_offset ?? null;
  } while (offset != null);
  if (!points.length) return { ok: true, points: 0, file: null };
  const date     = new Date().toISOString().slice(0, 10);
  const filePath = customPath ?? join(BACKUP_DIR, `atlas-backup-${date}.json`);
  await mkdir(BACKUP_DIR, { recursive: true });
  await writeFile(filePath, JSON.stringify({ version: '9.3.0', collection: COLLECTION, created_at: new Date().toISOString(), points }, null, 2), 'utf8');
  lastBackupTime = new Date().toISOString();
  logger?.info?.(`[atlas-memory] 备份：${points.length} 条记忆 → ${filePath}`);
  return { ok: true, points: points.length, file: filePath };
}

// ── Obsidian Bridge：主题聚类导出 ─────────────────────────────────────────────
async function runMirrorExport(logger) {
  if (!OBSIDIAN_VAULT) return { ok: false, reason: 'ATLAS_OBSIDIAN_VAULT 未配置' };

  // 1. 拉取全量 points（不需要向量，只要 payload）
  let offset = null;
  const allPoints = [];
  do {
    const body = { limit: 250, with_payload: true, with_vector: false };
    if (offset != null) body.offset = offset;
    const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', body);
    if (!r.ok) break;
    allPoints.push(...(r.body?.result?.points ?? []));
    offset = r.body?.result?.next_page_offset ?? null;
  } while (offset != null);

  if (!allPoints.length) return { ok: true, written: 0, total: 0, clusters: 0 };

  // 2. 按 memory_type + 主标签 聚类（主标签 = tags[0] 或 category 作为兜底）
  const clusters = new Map(); // "{memory_type}|{tag}" → points[]
  for (const pt of allPoints) {
    const memType    = pt.payload?.memory_type ?? 'fact';
    const primaryTag = pt.payload?.tags?.[0] ?? pt.payload?.category ?? 'general';
    const key        = `${memType}|${primaryTag}`;
    if (!clusters.has(key)) clusters.set(key, []);
    clusters.get(key).push(pt);
  }

  // 3. 预算文件名映射（供 wikilinks 引用）
  const clusterFilename = (key) => {
    const [memType, tag] = key.split('|');
    return `[${memType}] ${tag}`;
  };

  // 4. 构建标签重叠关系（用于 wikilinks：同一 point 带多个 tag → 相关聚类互联）
  const tagToKeys = new Map();
  for (const [key, points] of clusters) {
    for (const pt of points) {
      for (const t of (pt.payload?.tags ?? [])) {
        if (!tagToKeys.has(t)) tagToKeys.set(t, new Set());
        tagToKeys.get(t).add(key);
      }
    }
  }

  // 5. 全量覆写每个聚类文件
  const memoriesDir = join(OBSIDIAN_VAULT, OBSIDIAN_MIRROR_DIR, 'memories');
  await mkdir(memoriesDir, { recursive: true });

  let written = 0;
  for (const [key, points] of clusters) {
    const [memType, tag] = key.split('|');

    // 按重要性 DESC → hit_count DESC 排序
    points.sort((a, b) => {
      const ia = IMPORTANCE_ORDER[a.payload?.importance] ?? 2;
      const ib = IMPORTANCE_ORDER[b.payload?.importance] ?? 2;
      if (ib !== ia) return ib - ia;
      return (b.payload?.hit_count ?? 0) - (a.payload?.hit_count ?? 0);
    });

    // 相关聚类：共享 tag 的其他聚类（去重，限 MIRROR_LINK_MAX 个）
    const relatedKeys = new Set();
    for (const pt of points) {
      for (const t of (pt.payload?.tags ?? [])) {
        for (const rk of (tagToKeys.get(t) ?? [])) {
          if (rk !== key) relatedKeys.add(rk);
        }
      }
    }
    const relatedLinks = [...relatedKeys]
      .slice(0, MIRROR_LINK_MAX)
      .map(rk => `[[${clusterFilename(rk)}]]`)
      .join('  ');

    // 统计摘要
    const total    = points.length;
    const avgHits  = total ? Math.round(points.reduce((s, p) => s + (p.payload?.hit_count ?? 0), 0) / total) : 0;
    const impCounts = { critical: 0, high: 0, medium: 0, low: 0 };
    for (const pt of points) impCounts[pt.payload?.importance ?? 'medium']++;
    const now = new Date().toISOString();

    // 记忆列表（每条一节）
    const memoriesBody = points.map((pt, i) => {
      const imp   = pt.payload?.importance ?? 'medium';
      const hits  = pt.payload?.hit_count ?? 0;
      const date  = (pt.payload?.created_at ?? '').slice(0, 10);
      const src   = pt.payload?.source ?? '';
      const mtype = pt.payload?.memory_type ?? memType;
      const tags  = (pt.payload?.tags ?? []).map(t => `\`${t}\``).join(' ');
      return `### ${i + 1}. ${(pt.payload?.content ?? '').slice(0, 60)}${(pt.payload?.content ?? '').length > 60 ? '…' : ''}\n\n${pt.payload?.content ?? ''}\n\n*重要性: **${imp}** · 访问: ${hits}次 · 类型: ${mtype} · 来源: ${src} · ${date}*${tags ? `\n标签: ${tags}` : ''}\n`;
    }).join('\n---\n\n');

    const relatedSection = relatedLinks
      ? `\n## 关联主题\n\n${relatedLinks}\n`
      : '';

    const fileContent =
`---
memory_type: ${memType}
primary_tag: "${tag}"
total_memories: ${total}
avg_hit_count: ${avgHits}
critical_count: ${impCounts.critical}
high_count: ${impCounts.high}
medium_count: ${impCounts.medium}
low_count: ${impCounts.low}
last_updated: "${now}"
---

# [${memType}] ${tag}

> 共 **${total}** 条记忆 · 平均访问 **${avgHits}** 次 · 最后更新 ${now.slice(0, 16).replace('T', ' ')}

| 重要性 | 数量 |
|--------|------|
| 🔴 critical | ${impCounts.critical} |
| 🟠 high | ${impCounts.high} |
| 🟡 medium | ${impCounts.medium} |
| ⚪ low | ${impCounts.low} |

## 记忆内容

${memoriesBody}
${relatedSection}`;

    await writeFile(join(memoriesDir, `${clusterFilename(key)}.md`), fileContent, 'utf8');
    written++;
  }

  logger?.info?.(`[atlas-memory] Mirror 导出：${written} 个主题文件（${allPoints.length} 条记忆）→ ${memoriesDir}`);
  return { ok: true, written, total: allPoints.length, clusters: clusters.size, dir: memoriesDir };
}

// ── Obsidian Bridge：Dataview 仪表盘 ──────────────────────────────────────────
async function writeIndexDashboard() {
  if (!OBSIDIAN_VAULT) return;
  const mirrorDir = join(OBSIDIAN_VAULT, OBSIDIAN_MIRROR_DIR);
  await mkdir(mirrorDir, { recursive: true });

  const qdrantRes   = await httpReq(`${QDRANT}/collections/${COLLECTION}`);
  const totalPoints = qdrantRes.body?.result?.points_count ?? 0;
  const now         = new Date().toISOString();

  const content =
`---
type: atlas-dashboard
last_updated: "${now}"
---

# ATLAS Memory 监控台

> 向量库共 **${totalPoints}** 条记忆 · 最后刷新 ${now.slice(0, 16).replace('T', ' ')}
> 运行 \`atlas_obsidian_sync\` 重建聚类文件

---

## 高价值记忆 Top 10（按访问次数）

\`\`\`dataviewjs
const pages = dv.pages('"${OBSIDIAN_MIRROR_DIR}/memories"');
if (pages.length === 0) {
  dv.paragraph("⏳ 暂无数据，等待首次进化。请运行 \`atlas_obsidian_sync\` 工具。");
} else {
  dv.table(
    ["主题文件", "类型", "记忆数", "平均访问", "高价值数"],
    pages.sort(p => -(p.avg_hit_count ?? 0))
      .slice(0, 10)
      .map(p => [
        p.file.link,
        p.memory_type ?? "-",
        p.total_memories ?? 0,
        p.avg_hit_count ?? 0,
        (p.high_count ?? 0) + (p.critical_count ?? 0)
      ])
  );
}
\`\`\`

---

## 各主题记忆分布

\`\`\`dataviewjs
const pages = dv.pages('"${OBSIDIAN_MIRROR_DIR}/memories"');
if (pages.length === 0) {
  dv.paragraph("⏳ 暂无数据，等待首次进化。");
} else {
  dv.table(
    ["主题文件", "类型", "记忆总数", "🔴 critical", "🟠 high", "🟡 medium", "⚪ low"],
    pages.sort(p => -(p.total_memories ?? 0))
      .map(p => [
        p.file.link,
        p.memory_type ?? "-",
        p.total_memories ?? 0,
        p.critical_count ?? 0,
        p.high_count ?? 0,
        p.medium_count ?? 0,
        p.low_count ?? 0
      ])
  );
}
\`\`\`

---

## 近 7 天进化日志

\`\`\`dataviewjs
const logs = dv.pages('"${OBSIDIAN_MIRROR_DIR}/${EVOLUTION_LOG_SUBDIR}"')
  .sort(p => p.file.name, 'desc')
  .slice(0, 7);
if (logs.length === 0) {
  dv.paragraph("⏳ 尚无进化记录。记忆进化（CAPTURE / MERGE / PRUNE / UPGRADE）后自动生成。");
} else {
  dv.list(logs.map(p => p.file.link + " — " + p.file.name));
}
\`\`\`

---

*由 ATLAS Memory v9.4.0 · Obsidian Bridge 自动生成*
`;

  await writeFile(join(mirrorDir, '_index.md'), content, 'utf8');
}

// ── Phase 9：Obsidian 分层导出 + Git push ─────────────────────────────────────

async function gitPushVault(logger) {
  if (!OBSIDIAN_VAULT || !GITHUB_REPO) return { ok: false, reason: 'vault 或 GITHUB_REPO 未配置' };
  const { exec } = await import('child_process');
  const run = (cmd) => new Promise((resolve) => {
    exec(cmd, { cwd: OBSIDIAN_VAULT, timeout: 30_000 }, (err, stdout, stderr) => {
      resolve({ ok: !err, stdout: stdout?.trim(), stderr: stderr?.trim() });
    });
  });
  await run('git add -A');
  const date = new Date().toISOString().slice(0, 16).replace('T', ' ');
  const commit = await run(`git commit -m "atlas: auto-export ${date}" --allow-empty`);
  const push   = await run('git push origin main --quiet');
  if (!push.ok) logger?.warn?.(`[atlas-memory] git push 失败: ${push.stderr}`);
  return { ok: push.ok, committed: commit.stdout };
}

async function writeDomainIndex(domain, stats) {
  if (!OBSIDIAN_VAULT) return;
  if (domain.includes('×') || domain === 'None' || domain === 'null') return;
  const domainDir = await getCategoryForDomain(domain);
  const dir = join(OBSIDIAN_VAULT, 'L1', domainDir);
  await mkdir(dir, { recursive: true });
  const now = new Date().toISOString();
  const md = [
    '---',
    `domain: ${domain}`,
    `last_export: ${now}`,
    '---',
    '',
    `# ${domain} · 知识索引`,
    '',
    '```dataview',
    'TABLE level, topic, freshness_score as 新鲜度, created as 创建',
    `FROM "L1/${domainDir}"`,
    'WHERE file.name != "_index" AND level != null',
    'SORT level DESC, freshness_score DESC',
    '```',
    '',
    '## 知识层级（Qdrant）',
    `- L1 知识：${stats.l1} 条`,
    `- L2 洞见：${stats.l2} 条`,
    `- L3 智识：${stats.l3} 条`,
    `> L0 原料 ${stats.l0} 条仅存 Qdrant，Organize Agent 提炼后晋升为 L1 写入此目录`,
    '',
    '## 相关目录',
    `- [[L1/${domainDir}]]`,
    `- [[L2/${domainDir}]]`,
    `- [[L3/${domainDir}]]`,
  ].join('\n');
  await writeFile(join(dir, '_index.md'), md, 'utf8');
}

async function writeDomainMap(domainStats) {
  if (!OBSIDIAN_VAULT) return;
  const dir = join(OBSIDIAN_VAULT, '_系统');
  await mkdir(dir, { recursive: true });
  const now = new Date().toISOString();

  // 只在图谱表格里显示有 L1/L2/L3 内容的域（有链接），纯 L0 积压单独列出
  const activeDomains = domainStats.filter(d => d.l1 > 0 || d.l2 > 0 || d.l3 > 0)
    .sort((a, b) => (b.l1 + b.l2 + b.l3) - (a.l1 + a.l2 + a.l3));
  const pendingDomains = domainStats.filter(d => d.l1 === 0 && d.l2 === 0 && d.l3 === 0 && d.l0 > 0);

  const rows = activeDomains
    .map(d => `| [[L1/${DOMAIN_DIRS[d.domain] ?? d.domain}/_index\\|${d.domain}]] | ${d.l0} | ${d.l1} | ${d.l2} | ${d.l3} | ${((1 - (d.staleFraction ?? 0)) * 100).toFixed(0)}% |`)
    .join('\n');

  const pendingRows = pendingDomains.length
    ? [
        '',
        '## 待提炼（纯 L0，Organize Agent 处理后自动升级）',
        '| 域 | L0 积压 |',
        '|---|---|',
        ...pendingDomains.map(d => `| ${d.domain} | ${d.l0} |`),
      ].join('\n')
    : '';

  const md = [
    '---',
    `last_export: ${now}`,
    `type: domain-map`,
    '---',
    '',
    '# ATLAS 域图谱',
    `> 最后更新：${now.slice(0, 16).replace('T', ' ')}`,
    '',
    '| 域 | L0 | L1 | L2 | L3 | 新鲜度 |',
    '|---|---|---|---|---|---|',
    rows,
    pendingRows,
    '',
    '## 域连接图',
    ...activeDomains.map(d => `- [[L1/${DOMAIN_DIRS[d.domain] ?? d.domain}/_index|${d.domain}]]`),
  ].join('\n');
  await writeFile(join(dir, '域图谱.md'), md, 'utf8');
}

async function runLayeredExport(logger) {
  if (!OBSIDIAN_VAULT) return { ok: false, reason: 'ATLAS_OBSIDIAN_VAULT 未配置' };

  // 1. 拉取全量节点
  const allPoints = [];
  let offset = null;
  do {
    const body = { limit: 250, with_payload: true, with_vector: false,
      filter: { must_not: [{ key: 'status', match: { value: 'superseded' } }] } };
    if (offset != null) body.offset = offset;
    const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', body);
    if (!r.ok) break;
    allPoints.push(...(r.body?.result?.points ?? []));
    offset = r.body?.result?.next_page_offset ?? null;
  } while (offset != null);

  if (!allPoints.length) return { ok: true, domains: 0, files: 0 };

  // 2. 按域统计
  const byDomain = new Map();
  for (const pt of allPoints) {
    const d = pt.payload?.domain ?? '未分类';
    if (!byDomain.has(d)) byDomain.set(d, { l0:0, l1:0, l2:0, l3:0, staleFraction:0, staleCount:0, total:0 });
    const s = byDomain.get(d);
    const lvl = pt.payload?.level ?? 0;
    ['l0','l1','l2','l3'][lvl] && (s[['l0','l1','l2','l3'][lvl]]++);
    s.total++;
    if ((pt.payload?.freshness_score ?? 1.0) < FRESHNESS_REFRESH) s.staleCount++;
  }
  for (const [, s] of byDomain) s.staleFraction = s.total ? s.staleCount / s.total : 0;

  // 3. 为每个域写 _index.md（仅当域有 L1/L2/L3 内容时才写，纯 L0 积压不创建 Obsidian 目录）
  const domainStats = [];
  for (const [domain, stats] of byDomain) {
    if (stats.l1 === 0 && stats.l2 === 0 && stats.l3 === 0) {
      // 纯 L0 域：Organize Agent 尚未提炼，不在 Obsidian 创建目录，避免空壳目录误导
      domainStats.push({ domain, ...stats });
      continue;
    }
    await writeDomainIndex(domain, stats).catch(() => {});
    domainStats.push({ domain, ...stats });
  }

  // 3b. 清理损坏/孤儿目录：扫描 L1/ 和 L2/ 下实际存在的目录，删除无实际内容的
  try {
    const { readdir: rd, rm } = await import('fs/promises');
    // 有 L1+ 内容的域才算有效
    const validDirs = new Set(
      domainStats.filter(d => d.l1 > 0 || d.l2 > 0 || d.l3 > 0)
        .map(d => DOMAIN_DIRS[d.domain] ?? d.domain)
    );
    for (const prefix of ['L1', 'L2']) {
      const root = join(OBSIDIAN_VAULT, prefix);
      const actualDirs = await rd(root, { withFileTypes: true }).catch(() => []);
      for (const entry of actualDirs) {
        if (!entry.isDirectory()) continue;
        if (!validDirs.has(entry.name)) {
          const subFiles = await rd(join(root, entry.name)).catch(() => []);
          const hasContent = subFiles.some(f => f !== '_index.md' && f.endsWith?.('.md'));
          if (!hasContent) {
            await rm(join(root, entry.name), { recursive: true, force: true }).catch(() => {});
            logger?.info?.(`[atlas-memory] 分层导出: 清理空壳目录 ${prefix}/${entry.name}`);
          }
        }
      }
    }
  } catch {}

  // 4. 写全局域图谱
  await writeDomainMap(domainStats).catch(() => {});

  // 5. 保留旧版 Atlas_Mirror（标注已过时）
  const legacyNote = join(OBSIDIAN_VAULT, OBSIDIAN_MIRROR_DIR, '_DEPRECATED.md');
  await writeFile(legacyNote,
    '---\ntype: deprecated\n---\n\n> [!warning] 旧版导出（Atlas_Mirror）已停用\n> 请查看各域目录下的 `_index.md` 和 `_系统/域图谱.md`\n',
    'utf8'
  ).catch(() => {});

  // 6. Git push
  const pushResult = await gitPushVault(logger);
  logger?.info?.(`[atlas-memory] 分层导出: ${domainStats.length}域, git=${pushResult.ok}`);

  return { ok: true, domains: domainStats.length, files: domainStats.length + 1, pushed: pushResult.ok };
}

// ── Obsidian Bridge：每日进化日志 ─────────────────────────────────────────────
async function appendEvolutionLog(type, message) {
  if (!OBSIDIAN_VAULT) return;
  const today   = new Date().toISOString().slice(0, 10);
  const logDir  = join(OBSIDIAN_VAULT, OBSIDIAN_MIRROR_DIR);
  const logFile = join(logDir, `${today}.md`);
  const time    = new Date().toTimeString().slice(0, 5);
  try {
    await mkdir(logDir, { recursive: true });
    // 新文件：写入日志头
    let needsHeader = false;
    try { await readFile(logFile, 'utf8'); } catch { needsHeader = true; }
    if (needsHeader) {
      await writeFile(logFile,
        `---\ndate: ${today}\ntype: evolution-log\n---\n\n# 进化日志 ${today}\n\n`, 'utf8');
    }
    await appendFile(logFile, `- \`${time}\` **[${type}]** ${message}\n`, 'utf8');
  } catch { /* 静默：日志写失败不影响主流程 */ }
}

// ── v10 Schema 迁移（启动时运行，向后兼容旧记录）─────────────────────────────
async function migrateSchema(logger) {
  let offset = null;
  let patched = 0;
  const now = new Date().toISOString();
  do {
    const body = { limit: 250, with_payload: true, with_vector: false };
    if (offset != null) body.offset = offset;
    const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', body);
    if (!r.ok) break;
    const points = r.body?.result?.points ?? [];
    for (const pt of points) {
      const p = pt.payload ?? {};
      if (p.level !== undefined && p.freshness_score !== undefined) continue;
      const patch = {};
      if (p.level            === undefined) patch.level            = LEVEL_KNOWLEDGE;
      if (p.domain           === undefined) patch.domain           = null;
      if (p.topic            === undefined) patch.topic            = p.tags?.[0] ?? p.domain ?? null;
      if (p.freshness_score  === undefined) patch.freshness_score  = 1.0;
      if (p.decay_rate       === undefined) patch.decay_rate       = 'medium';
      if (p.last_verified    === undefined) patch.last_verified    = now;
      if (p.source_ids       === undefined) patch.source_ids       = [];
      if (p.associated_ids   === undefined) patch.associated_ids   = [];
      if (p.derived_to_id    === undefined) patch.derived_to_id    = null;
      if (p.obsidian_path    === undefined) patch.obsidian_path    = null;
      if (p.acquisition_source === undefined) patch.acquisition_source = p.source ?? 'auto-capture';
      await qdrantPatchPayload(pt.id, patch);
      patched++;
    }
    offset = r.body?.result?.next_page_offset ?? null;
  } while (offset != null);
  if (patched > 0) logger?.info?.(`[atlas-memory] v10 schema 迁移：${patched} 条记录已更新`);
  return { patched };
}

// ── 启动时还原动态域（防重启丢失）────────────────────────────────────────────
async function restoreDynamicDomains(logger) {
  // Scroll all points, collect distinct non-null domain values not in static DOMAIN_DIRS
  const seen = new Set();
  let offset = null;
  do {
    const body = { limit: 250, with_payload: true, with_vector: false,
      filter: { must_not: [{ is_null: { key: 'domain' } }] } };
    if (offset != null) body.offset = offset;
    const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', body);
    if (!r.ok) break;
    for (const pt of r.body?.result?.points ?? []) {
      const d = pt.payload?.domain;
      if (d && !DOMAIN_DIRS[d]) seen.add(d);
    }
    offset = r.body?.result?.next_page_offset ?? null;
  } while (offset != null);

  if (!seen.size) return;

  for (const domainName of seen) {
    DOMAIN_DIRS[domainName] = domainName;
    // Read description from _维度图谱.md if it exists
    let desc = domainName;
    if (OBSIDIAN_VAULT) {
      const mapPath = join(OBSIDIAN_VAULT, 'L1', domainName, '_维度图谱.md');
      const raw = await readFile(mapPath, 'utf8').catch(() => null);
      if (raw) {
        const m = raw.match(/^description:\s*(.+)$/m);
        if (m) desc = m[1].trim();
      }
    }
    DOMAIN_DESCRIPTIONS[domainName] = desc;
    // Pre-warm embedding cache
    const vec = await embed(desc);
    if (vec) domainEmbeddingCache.set(domainName, vec);
    logger?.info?.(`[atlas-memory] 还原动态域: "${domainName}"`);
  }
}

// ── v10 L0 原料统一摄入 ───────────────────────────────────────────────────────
async function intakeToL0({ content, domain, topic, source = 'manual', tags = [], category = 'work', importance = 'medium', memory_type = 'fact', sessionKey, knowledge_type = 'capture',
  group_id, chunk_index, group_total, content_type = 'other', source_meta = null }) {
  return writeQueue.push(WRITE_PRIORITY.CAPTURE, async () => {
    const domainDir = DOMAIN_DIRS[domain] ?? null;
    // L0 原料只存 Qdrant，不写 Obsidian——Obsidian 是精炼后知识的视图，由 Organize Agent 晋升 L1 时写入
    let   obsidianPath = null;

    const vector = await embed(content);
    if (!vector) return { ok: false, error: 'embed failed' };
    await ensureCollection();
    const now = new Date().toISOString();
    // v11: source_type 自动检测
    const source_type      = detectSourceType(content, '', Array.isArray(tags) ? tags : []);
    const knowledge_purpose = detectKnowledgePurpose(content, Array.isArray(tags) ? tags : [], source_type);
    const expires_at       = calcTTLExpiry(source_type);
    const decay_rate       = inferDecayRate(domain, tags, source_type);
    const payload = {
      content:            content.trim(),
      title:              topic ?? tags[0] ?? domain ?? '',
      category,
      importance,
      tags:               Array.isArray(tags) ? tags : [],
      memory_type,
      created_at:         now,
      source,
      session_key:        sessionKey ?? 'manual',
      hit_count:          0,
      last_accessed_at:   null,
      status:             'active',
      feedback_score:     1.0,
      level:              LEVEL_RAW,
      domain:             domain ?? null,
      topic:              topic ?? tags[0] ?? domain ?? null,
      freshness_score:    1.0,
      decay_rate,
      last_verified:      now,
      source_ids:         [],
      associated_ids:     [],
      derived_to_id:      null,
      obsidian_path:      obsidianPath,
      acquisition_source: source,
      knowledge_type,
      completeness_score: null,
      completeness_gaps:  [],
      domain_score:       null,  // 由整理Agent晋升L1时填入
      // v11 新增字段
      source_type,
      knowledge_purpose,
      expires_at,
      platform:           source_meta?.platform ?? null,
      // v12 新增字段
      group_id:           group_id    ?? null,
      chunk_index:        chunk_index ?? null,
      group_total:        group_total ?? null,
      content_type:       content_type ?? 'other',
      source_meta:        source_meta  ?? null,
      record_type:        RECORD_TYPES.KNOWLEDGE,
      confidence:         CONFIDENCE_DEFAULT,
      entity_ids:         [],
      relation_ids:       [],
    };
    // 分片信号自动检测
    if (!payload.group_id && FRAGMENT_SIGNALS.some(p => p.test(content))) {
      payload.group_id = `grp_${Date.now().toString(36)}`;
    }
    return upsert(vector, payload);
  });
}

// ── Phase 3：整理Agent（L0→L1）────────────────────────────────────────────────

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

async function getDomainEmbeddings() {
  const total = Object.keys(DOMAIN_DESCRIPTIONS).length;
  if (domainEmbeddingCache.size >= total) return domainEmbeddingCache;
  for (const [domain, desc] of Object.entries(DOMAIN_DESCRIPTIONS)) {
    if (domainEmbeddingCache.has(domain)) continue;
    const vec = await embed(desc);
    if (vec) domainEmbeddingCache.set(domain, vec);
  }
  return domainEmbeddingCache;
}

async function matchDomainForVector(vector) {
  const cache = await getDomainEmbeddings();
  let best = null, bestScore = 0;
  for (const [domain, domVec] of cache) {
    const score = cosine(vector, domVec);
    if (score > bestScore) { bestScore = score; best = domain; }
  }
  if (bestScore >= DOMAIN_MATCH_SCORE)    return { domain: best, score: bestScore };
  if (bestScore >= DOMAIN_SUBDOMAIN_SCORE) return { domain: best, score: bestScore, weak: true };
  return { domain: null, score: bestScore };
}

// ── v12 extractL1Content（DeepSeek only · 多节点输出 · 实体+关系提取）────────
async function extractL1Content(content, domain, contentType = 'other') {
  const systemPrompt = `你是知识提炼专家。从原始内容中提取结构化知识节点、实体和关系。
严格规则：
1. 只提取原文中确实存在的信息，不推测不编造
2. 每个知识节点必须自包含，读者无需其他节点即可理解
3. 每个节点内容不超过5000字（超过则拆分为多个节点，保持逻辑完整）
4. 实体：跨域可复用的核心概念（如"钩子句"、"止损位"、"完播率"）
5. 关系：节点间的逻辑联系
严格只输出有效JSON，不要任何解释或markdown代码块`;

  const typeHint = {
    video_script:    '"hook":"","structure":"","pain_points":[],"cta":"","applicable_scenarios":[]',
    trading_signal:  '"entry":"","stop_loss":"","take_profit":"","position_size":"","reasoning":"","timeframe":""',
    sop:             '"steps":[],"preconditions":[],"expected_outcome":"","tools_required":[]',
    process:         '"steps":[],"preconditions":[],"expected_outcome":"","tools_required":[]',
  }[contentType] ?? '"key_points":[],"applicable_scenarios":[],"examples":[]';

  const userPrompt = `请从以下内容中提取知识节点、实体和关系。
领域：${domain ?? '未知'}  内容类型：${contentType}

原始内容：
${content}

输出格式（严格JSON）：
{
  "nodes": [
    {
      "title": "节点标题（8字以内）",
      "summary": "核心摘要（3-5句）",
      "content": "完整知识内容（不超过5000字，若内容过长请拆分为多个nodes，每个保持逻辑完整）。针对${contentType}类型，请包含：${typeHint}",
      "content_type": "${contentType}",
      "knowledge_purpose": "understanding|production|process",
      "tags": ["标签1","标签2"],
      "faithfulness_score": 0.85
    }
  ],
  "entities": [
    { "canonical_name":"实体规范名称","aliases":["别名"],"definition":"简短定义（20-50字）","domains":["适用域"] }
  ],
  "relations": [
    { "source_title":"节点A的title","target_title":"节点B的title","relation_type":"supports|contradicts|extends|depends_on|used_in|evolved_from|cross_domain","strength":0.7,"context":"关系说明" }
  ]
}`;

  const raw = await deepseekGenerate(systemPrompt, userPrompt, 6000);
  if (!raw) return null;
  const result = parseJsonObject(raw);
  if (!result || !Array.isArray(result.nodes) || result.nodes.length === 0) return null;
  for (const node of result.nodes) {
    if (!node.title || !node.summary) return null;
    if (node.content?.length > EMBED_SAFE_CHARS) {
      console.warn(`[extractL1Content] node "${node.title}" content ${node.content.length} chars exceeds safe limit`);
    }
  }
  if (!Array.isArray(result.entities))  result.entities  = [];
  if (!Array.isArray(result.relations)) result.relations = [];
  return result;
}

// 按内容类型计算完整度（相对评分，不跨类型比较）
function calcCompleteness(node, contentType) {
  const universalScore = (
    (node.summary?.trim()              ? 0.15 : 0) +
    ((node.key_points ?? []).length > 0 ? 0.20 : 0) +
    ((node.tags ?? []).length > 0       ? 0.05 : 0)
  ); // 最高 0.40

  const typeChecks = {
    concept:   [
      [node.definition?.trim(),                             0.30],
      [node.scope?.trim(),                                  0.20],
      [(node.examples ?? []).length > 0,                   0.25],
      [(node.related_concepts ?? []).length > 0,           0.25],
    ],
    argument:  [
      [node.claim?.trim(),                                  0.20],
      [node.reasoning?.trim(),                             0.30],
      [(node.evidence ?? []).filter(Boolean).length > 0,  0.30],
      [node.limitations?.trim(),                           0.20],
    ],
    procedure: [
      [(node.steps ?? []).length > 0,                      0.35],
      [node.preconditions?.trim(),                         0.20],
      [node.expected_outcome?.trim(),                      0.25],
      [node.edge_cases?.trim(),                            0.20],
    ],
    fact:      [
      [node.statement?.trim(),                             0.35],
      [node.source_context?.trim(),                        0.25],
      [node.temporal_scope?.trim(),                        0.20],
      [node.confidence?.trim(),                            0.20],
    ],
    principle: [
      [node.rule_statement?.trim(),                        0.25],
      [node.rationale?.trim(),                             0.20],
      [node.applicable_scenarios?.trim(),                  0.25],
      [node.exceptions?.trim(),                            0.15],
      [(node.examples ?? []).length > 0,                   0.15],
    ],
    course: [
      [node.rule_statement?.trim(),                        0.25],
      [node.applicable_scenarios?.trim(),                  0.25],
      [(node.steps ?? []).length > 0,                      0.20],
      [node.exceptions?.trim(),                            0.15],
      [(node.examples ?? []).length > 0,                   0.15],
    ],
    book: [
      [node.rule_statement?.trim(),                        0.30],
      [node.rationale?.trim(),                             0.20],
      [node.applicable_scenarios?.trim(),                  0.25],
      [(node.examples ?? []).length > 0,                   0.25],
    ],
    note: [
      [node.rule_statement?.trim(),                        0.30],
      [node.rationale?.trim(),                             0.25],
      [node.applicable_scenarios?.trim(),                  0.25],
      [node.exceptions?.trim(),                            0.20],
    ],
    article: [
      [node.claim?.trim() || node.rule_statement?.trim(),  0.25],
      [node.reasoning?.trim() || node.rationale?.trim(),   0.30],
      [(node.evidence ?? []).filter(Boolean).length > 0,   0.25],
      [node.limitations?.trim() || node.exceptions?.trim(),0.20],
    ],
    video_script: [
      [node.hook?.trim(),                                  0.30],
      [node.structure?.trim(),                             0.25],
      [(node.pain_points ?? []).length > 0,               0.25],
      [node.cta?.trim(),                                   0.20],
    ],
    sop: [
      [(node.steps ?? []).length > 0,                      0.40],
      [node.preconditions?.trim(),                         0.20],
      [node.expected_outcome?.trim(),                      0.25],
      [Array.isArray(node.tools_required) && node.tools_required.length > 0, 0.15],
    ],
  };

  const checks = typeChecks[contentType] ?? typeChecks.principle;
  let typeScore = 0;
  const gaps = [];
  const typeFieldNames = {
    concept:      ['definition','scope','examples','related_concepts'],
    argument:     ['claim','reasoning','evidence','limitations'],
    procedure:    ['steps','preconditions','expected_outcome','edge_cases'],
    fact:         ['statement','source_context','temporal_scope','confidence'],
    principle:    ['rule_statement','rationale','applicable_scenarios','exceptions','examples'],
    course:       ['rule_statement','applicable_scenarios','steps','exceptions','examples'],
    book:         ['rule_statement','rationale','applicable_scenarios','examples'],
    note:         ['rule_statement','rationale','applicable_scenarios','exceptions'],
    article:      ['claim','reasoning','evidence','limitations'],
    video_script: ['hook','structure','pain_points','cta'],
    sop:          ['steps','preconditions','expected_outcome','tools_required'],
  };
  const fieldNames = typeFieldNames[contentType] ?? typeFieldNames.principle;
  checks.forEach(([ok, w], i) => {
    if (ok) typeScore += w; else gaps.push(fieldNames[i]);
  });
  if (!node.summary?.trim()) gaps.push('summary');

  const score = universalScore + typeScore * 0.60;
  return { score: Math.min(score, 1.0), gaps };
}

async function writeL1Obsidian(domain, topic, l1Data, sourceL0Path) {
  if (!OBSIDIAN_VAULT) return null;
  const category  = await getCategoryForDomain(domain);
  const domainDir = category;  // use category as folder, not raw domain
  const l1Dir     = join(OBSIDIAN_VAULT, 'L1', domainDir);
  await mkdir(l1Dir, { recursive: true });
  const slug     = topic.replace(/[/\\:*?"<>|]/g, '-').slice(0, 50);
  const filename = `${slug}.md`;
  const ct = l1Data.content_type ?? 'principle';

  // 按内容类型生成对应区块
  const typeBlocks = {
    concept: [
      '## 定义', l1Data.definition ?? '',
      '', '## 适用范围', l1Data.scope ?? '',
      '', '## 具体例子', ...(l1Data.examples ?? []).map(e => `- ${typeof e === 'string' ? e : (e.text ?? e.example ?? e.content ?? JSON.stringify(e))}`),
      '', '## 关联概念', ...(l1Data.related_concepts ?? []).map(c => `- ${typeof c === 'string' ? c : (c.name ?? JSON.stringify(c))}`),
    ],
    argument: [
      '## 核心论点', l1Data.claim ?? '',
      '', '## 推理链条', l1Data.reasoning ?? '',
      '', '## 支撑证据', ...(l1Data.evidence ?? []).map(e => `- ${typeof e === 'string' ? e : (e.text ?? e.fact ?? JSON.stringify(e))}`),
      '', '## 局限性', l1Data.limitations ?? '',
      '', '## 反驳与反例', l1Data.counter_evidence ?? '',
    ],
    procedure: [
      '## 操作步骤', ...(l1Data.steps ?? []).map((s, i) => `${i + 1}. ${typeof s === 'string' ? s : (s.step ?? JSON.stringify(s))}`),
      '', '## 前提条件', l1Data.preconditions ?? '',
      '', '## 预期结果', l1Data.expected_outcome ?? '',
      '', '## 边界情况', l1Data.edge_cases ?? '',
    ],
    fact: [
      '## 事实陈述', l1Data.statement ?? '',
      '', '## 来源背景', l1Data.source_context ?? '',
      '', '## 时效性', l1Data.temporal_scope ?? '',
      '', '## 可信度', l1Data.confidence ?? '',
    ],
    principle: [
      '## 原则表述', l1Data.rule_statement ?? '',
      '', '## 底层逻辑', l1Data.rationale ?? '',
      '', '## 适用场景', l1Data.applicable_scenarios ?? '',
      '', '## 例外情况', l1Data.exceptions ?? '',
      ...(l1Data.examples?.length ? ['', '## 具体例子', ...(l1Data.examples ?? []).map(e => `- ${typeof e === 'string' ? e : (e.text ?? e.example ?? JSON.stringify(e))}`)] : []),
    ],
    course: [
      '## 核心原理', l1Data.rule_statement ?? l1Data.rationale ?? '',
      '', '## 适用场景', l1Data.applicable_scenarios ?? l1Data.scope ?? '',
      '', '## 操作要点', Array.isArray(l1Data.steps) ? l1Data.steps.map((s, i) => `${i+1}. ${typeof s === 'string' ? s : (s.step ?? JSON.stringify(s))}`).join('\n') : (l1Data.steps ?? ''),
      '', '## 注意事项', l1Data.exceptions ?? l1Data.edge_cases ?? '',
      ...(l1Data.examples?.length ? ['', '## 课程案例', ...(l1Data.examples ?? []).map(e => `- ${typeof e === 'string' ? e : (e.text ?? e.example ?? JSON.stringify(e))}`)] : []),
    ],
    book: [
      '## 核心主张', l1Data.rule_statement ?? '',
      '', '## 底层逻辑', l1Data.rationale ?? '',
      '', '## 可执行要点', ...(l1Data.key_points ?? []).map(p => `- ${typeof p === 'string' ? p : JSON.stringify(p)}`),
      '', '## 适用场景', l1Data.applicable_scenarios ?? '',
      ...(l1Data.examples?.length ? ['', '## 书中案例', ...(l1Data.examples ?? []).map(e => `- ${typeof e === 'string' ? e : (e.text ?? e.example ?? JSON.stringify(e))}`)] : []),
    ],
    note: [
      '## 核心观点', l1Data.rule_statement ?? '',
      '', '## 关键洞见', l1Data.rationale ?? '',
      '', '## 适用场景', l1Data.applicable_scenarios ?? '',
      '', '## 例外情况', l1Data.exceptions ?? '',
      ...(l1Data.examples?.length ? ['', '## 具体例子', ...(l1Data.examples ?? []).map(e => `- ${typeof e === 'string' ? e : (e.text ?? e.example ?? JSON.stringify(e))}`)] : []),
    ],
    article: [
      '## 核心论点', l1Data.claim ?? l1Data.rule_statement ?? '',
      '', '## 推理链条', l1Data.reasoning ?? l1Data.rationale ?? '',
      '', '## 支撑证据', ...(l1Data.evidence ?? []).map(e => `- ${typeof e === 'string' ? e : (e.text ?? e.fact ?? JSON.stringify(e))}`),
      '', '## 局限性', l1Data.limitations ?? l1Data.exceptions ?? '',
      ...(l1Data.examples?.length ? ['', '## 具体例子', ...(l1Data.examples ?? []).map(e => `- ${typeof e === 'string' ? e : (e.text ?? e.example ?? JSON.stringify(e))}`)] : []),
    ],
    video_script: [
      '## 钩子/开场', l1Data.hook ?? '',
      '', '## 内容结构', l1Data.structure ?? '',
      '', '## 痛点/价值点', Array.isArray(l1Data.pain_points) ? l1Data.pain_points.map(p => `- ${typeof p === 'string' ? p : JSON.stringify(p)}`).join('\n') : (l1Data.pain_points ?? ''),
      '', '## 行动呼吁(CTA)', l1Data.cta ?? '',
      '', '## 适用场景', Array.isArray(l1Data.applicable_scenarios) ? l1Data.applicable_scenarios.join('；') : (l1Data.applicable_scenarios ?? ''),
    ],
    sop: [
      '## 操作步骤', ...(l1Data.steps ?? []).map((s, i) => `${i + 1}. ${typeof s === 'string' ? s : (s.step ?? JSON.stringify(s))}`),
      '', '## 所需工具', Array.isArray(l1Data.tools_required) ? l1Data.tools_required.map(t => `- ${t}`).join('\n') : (l1Data.tools_required ?? ''),
      '', '## 前提条件', l1Data.preconditions ?? '',
      '', '## 预期结果', l1Data.expected_outcome ?? '',
    ],
  };

  const faithfulness = l1Data.faithfulness_score ?? 1.0;
  const hallucinationFlag = faithfulness < 0.6 ? ' ⚠️幻觉风险' : '';
  const lines = [
    '---',
    `level: L1`,
    `content_type: ${ct}`,
    `domain: ${domain ?? '未分类'}`,
    `topic: ${topic}`,
    `source_l0: ${sourceL0Path ?? ''}`,
    `created: ${new Date().toISOString()}`,
    `tags: [${(l1Data.tags ?? []).map(t => `"${t}"`).join(', ')}]`,
    `completeness_score: ${l1Data.completeness_score ?? 0}`,
    `completeness_gaps: [${(l1Data.completeness_gaps ?? []).map(g => `"${g}"`).join(', ')}]`,
    `faithfulness_score: ${faithfulness}`,
    '---',
    '',
    `# ${l1Data.title ?? topic}`,
    `> 类型：${ct}  完整度：${((l1Data.completeness_score ?? 0) * 100).toFixed(0)}%${hallucinationFlag}`,
    '',
    '## 摘要',
    l1Data.summary ?? '',
    '',
    '## 核心要点',
    ...(l1Data.key_points ?? []).map(p => `- ${p}`),
    '',
    ...(typeBlocks[ct] ?? typeBlocks.principle),
    '',
  ];
  await writeFile(join(l1Dir, filename), lines.join('\n'), 'utf8');
  return `L1/${domainDir}/${filename}`;
}

async function runOrganizeAgent(logger) {
  const log = logger ?? console;
  const scrollR = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', {
    filter: { must: [{ key: 'level', match: { value: LEVEL_RAW } }] },
    limit: ORGANIZE_BATCH_MAX, with_payload: true, with_vector: true,
  });
  if (!scrollR.ok) return { processed: 0, promoted: 0, skipped: 0 };

  const l0Points = scrollR.body?.result?.points ?? [];
  if (!l0Points.length) {
    log.debug?.('[atlas-memory] 整理Agent: 无L0待处理记录');
    return { processed: 0, promoted: 0, skipped: 0 };
  }
  log.info?.(`[atlas-memory] 整理Agent: 处理 ${l0Points.length} 条L0记录`);
  let promoted = 0, skipped = 0;

  // 无效内容特征：报错信息、API失败提示等被误存为L0原料
  const INVALID_CONTENT_PATTERNS = [
    /Access denied/i,
    /无法生成笔记/,
    /不包含实际.*课程/,
    /全部是系统接口报错/,
    /请.*重新粘贴|请.*上传完整/,
    /转写内容.*错误|录音转写.*失败/,
  ];

  for (const pt of l0Points) {
    const content = pt.payload?.content;
    if (!content?.trim()) { skipped++; continue; }

    // 过滤无效内容（报错信息、API失败提示）
    if (INVALID_CONTENT_PATTERNS.some(p => p.test(content))) {
      // 直接删除该L0脏记录，避免反复处理
      await writeQueue.push(WRITE_PRIORITY.AGENT, () => qdrantDelete([pt.id])).catch(() => {});
      skipped++;
      continue;
    }

    // 域匹配
    const storedDomain = pt.payload?.domain;
    let domain, matchScore = 1.0;
    if (storedDomain && DOMAIN_DESCRIPTIONS[storedDomain]) {
      domain = storedDomain;
    } else if (pt.vector) {
      const match = await matchDomainForVector(pt.vector).catch(() => null);
      domain = match?.domain ?? null;
      matchScore = match?.score ?? 0;
    }

    // DeepSeek 提取（多节点 + 实体 + 关系）
    const contentType = pt.payload?.content_type ?? 'other';
    let result;
    try { result = await extractL1Content(content, domain, contentType); } catch (e) {
      log.warn?.(`[atlas-memory] extractL1Content failed for ${pt.id}:`, e?.message);
    }
    if (!result) { skipped++; continue; }

    const { nodes, entities, relations } = result;
    const now = new Date().toISOString();
    const nodeIds = [];

    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i];
      const nodeText = (node.content || node.summary).slice(0, EMBED_SAFE_CHARS);
      // i===0：patch 原 L0 记录 payload（不更新向量），跳过 embed 省 ~40s
      const nodeVector = i === 0 ? (pt.vector ?? null) : await embed(nodeText).catch(() => null);
      if (!nodeVector) { nodeIds.push(null); continue; }

      const mergedTags = [...new Set([...(node.tags ?? []), ...(pt.payload?.tags ?? [])])];
      const nodeContentType = node.content_type ?? pt.payload?.content_type ?? 'other';
      const { score: initScore, gaps: initGaps } = calcCompleteness(node, nodeContentType);
      const nodePayload = {
        level: LEVEL_KNOWLEDGE, domain: domain ?? '未分类', topic: node.title, title: node.title ?? '',
        status: 'active',
        content: nodeText, tags: mergedTags, summary: node.summary,
        knowledge_purpose: node.knowledge_purpose ?? 'understanding',
        faithfulness_score: node.faithfulness_score ?? 1.0,
        hallucination_risk: (node.faithfulness_score ?? 1.0) < 0.6,
        confidence: CONFIDENCE_DEFAULT, record_type: RECORD_TYPES.KNOWLEDGE,
        entity_ids: [], relation_ids: [], last_verified: now, freshness_score: 1.0,
        decay_rate: inferDecayRate(domain, mergedTags, pt.payload?.source_type),
        source_type: pt.payload?.source_type, expires_at: pt.payload?.expires_at,
        group_id: pt.payload?.group_id ?? null, domain_score: parseFloat(matchScore.toFixed(3)),
        obsidian_path: null,  // 先置null，writeL1Obsidian完成后再patch正确路径
        content_type: nodeContentType,
        completeness_score: parseFloat(initScore.toFixed(2)),
        completeness_gaps: initGaps,
      };

      if (i === 0) {
        await writeQueue.push(WRITE_PRIORITY.AGENT, () => qdrantPatchPayload(pt.id, nodePayload));
        nodeIds.push(pt.id);
        // 晋升为 L1：写 Obsidian，并删除旧的 L0 磁盘文件（如有）
        const sourceL0Path = pt.payload?.obsidian_path;
        if (sourceL0Path?.startsWith('L0/') && OBSIDIAN_VAULT) {
          unlink(join(OBSIDIAN_VAULT, sourceL0Path)).catch(() => {});
        }
        const l1Path = await writeL1Obsidian(domain, node.title, node, sourceL0Path).catch(() => null);
        if (l1Path) {
          await writeQueue.push(WRITE_PRIORITY.AGENT, () => qdrantPatchPayload(pt.id, { obsidian_path: l1Path }));
        }
      } else {
        const newId = stableId(node.title + String(pt.id) + String(i));
        await writeQueue.push(WRITE_PRIORITY.AGENT, () =>
          httpReq(`${QDRANT}/collections/${COLLECTION}/points?wait=true`, 'PUT', {
            points: [{ id: newId, vector: nodeVector, payload: nodePayload }],
          })
        );
        nodeIds.push(newId);
        // 多节点也需要写 Obsidian 文件并回写路径
        const l1PathNew = await writeL1Obsidian(domain, node.title, node, null).catch(() => null);
        if (l1PathNew) {
          await writeQueue.push(WRITE_PRIORITY.AGENT, () => qdrantPatchPayload(newId, { obsidian_path: l1PathNew }));
        }
      }
    }

    // 实体注册
    const entityIds = [];
    for (const ent of entities) {
      try {
        const er = await upsertEntity({ canonical_name: ent.canonical_name, aliases: ent.aliases ?? [],
          domains: ent.domains ?? [domain].filter(Boolean), definition: ent.definition ?? '' });
        if (er.ok && er.entity_id != null) entityIds.push(er.entity_id);
      } catch (e) { log.warn?.(`[atlas-memory] upsertEntity failed: ${ent.canonical_name}`, e?.message); }
    }
    if (entityIds.length && nodeIds[0] != null) {
      await writeQueue.push(WRITE_PRIORITY.AGENT, () => qdrantPatchPayload(nodeIds[0], { entity_ids: entityIds }));
    }

    // 关系写入
    const titleToId = {};
    nodes.forEach((n, i) => { if (nodeIds[i] != null) titleToId[n.title] = nodeIds[i]; });
    for (const rel of relations) {
      const srcId = titleToId[rel.source_title];
      const tgtId = titleToId[rel.target_title];
      if (!srcId || !tgtId) continue;
      const rtype = Object.values(RELATION_TYPES).includes(rel.relation_type) ? rel.relation_type : RELATION_TYPES.SUPPORTS;
      try {
        await upsertRelation({ source_id: srcId, target_id: tgtId, relation_type: rtype,
          strength: rel.strength ?? 0.7, context: rel.context ?? '' });
      } catch (e) { log.warn?.(`[atlas-memory] upsertRelation failed`, e?.message); }
    }

    promoted++;
    appendEvolutionLog('ORGANIZE', `L0→L1: [${domain ?? '未分类'}] ${nodes.length}节点 ${entities.length}实体`).catch(() => {});
  }

  log.info?.(`[atlas-memory] 整理Agent: 晋升${promoted}条, 跳过${skipped}条`);
  return { processed: l0Points.length, promoted, skipped };
}

// ── Phase 4：域检测Agent ──────────────────────────────────────────────────────

function centroid(vectors) {
  if (!vectors.length) return null;
  const dim = vectors[0].length;
  const sum = new Array(dim).fill(0);
  for (const v of vectors) for (let i = 0; i < dim; i++) sum[i] += v[i];
  const len = vectors.length;
  return sum.map(x => x / len);
}

function clusterNodes(nodes, minSim = CLUSTER_MIN_SCORE, minSize = CLUSTER_MIN_SIZE) {
  // Greedy threshold clustering: first unclustered node becomes seed
  const clusters = [];
  const assigned = new Set();

  for (let i = 0; i < nodes.length; i++) {
    if (assigned.has(i)) continue;
    const cluster = [i];
    assigned.add(i);
    for (let j = i + 1; j < nodes.length; j++) {
      if (assigned.has(j)) continue;
      if (cosine(nodes[i].vector, nodes[j].vector) >= minSim) {
        cluster.push(j);
        assigned.add(j);
      }
    }
    if (cluster.length >= minSize) clusters.push(cluster.map(idx => nodes[idx]));
  }
  return clusters;
}

async function inferNewDomain(samples) {
  const excerpts = samples.slice(0, 5).map((s, i) => `${i + 1}. ${s.payload?.content?.slice(0, 120) ?? ''}`).join('\n');
  const sys = '你是知识分类专家。严格输出JSON，不要解释，不要markdown代码块。';
  const user =
    `以下是同一知识簇中的记忆样本：\n${excerpts}\n\n` +
    `请推断这个知识簇属于什么知识领域。\n` +
    `【命名规则】域名必须是知识类型（如：情感学、营销、战略、储能电池），` +
    `禁止使用人名（如刘克亚、科特勒）、课程编号（如02-营销学）、平台名称、来源标签；不超过8个字。\n` +
    `输出JSON：\n` +
    `{"domain_name":"知识领域名（≤8字，不含人名/编号）","description":"一句话描述（15-30字）",` +
    `"is_valid_domain":true,"dimensions":["维度1","维度2","维度3"],"keywords":["关键词1","关键词2","关键词3","关键词4","关键词5"]}`;

  let raw = await deepseekGenerate(sys, user, 400);
  if (!raw) {
    raw = await omlxGenerate(sys, user.slice(0, 1200), 400, AGENT_OMLX_TIMEOUT_MS);
    if (!raw) return null;
  }
  try {
    const cleaned = raw.replace(/```[a-z]*\n?/gi, '').replace(/```/g, '').trim();
    const parsed = JSON.parse(cleaned);
    if (!parsed.domain_name || !parsed.description) return null;
    if (parsed.is_valid_domain === false) return null; // LLM 认为不是合法知识域
    // 硬校验：拒绝含数字编号或人名格式的域名
    if (/\d/.test(parsed.domain_name)) return null;
    if (/^[一-龥]{1,3}(学|法|课)?$/.test(parsed.domain_name) &&
        /[A-Za-z]/.test(parsed.domain_name)) return null;
    parsed.dimensions = Array.isArray(parsed.dimensions) ? parsed.dimensions.slice(0, 5) : [];
    parsed.keywords   = Array.isArray(parsed.keywords)   ? parsed.keywords.slice(0, 8)   : [];
    return parsed;
  } catch {
    return null;
  }
}

async function createDomainStructure(domainName, domainInfo) {
  if (!OBSIDIAN_VAULT) return null;
  const category = await getCategoryForDomain(domainName);
  for (const level of ['L1', 'L2', 'L3']) {
    const dir = join(OBSIDIAN_VAULT, level, category);
    await mkdir(dir, { recursive: true });
  }
  const domainDir = join(OBSIDIAN_VAULT, 'L1', category);

  // Write dimension map
  const dims  = (domainInfo.dimensions ?? []).map(d => `- ${d}`).join('\n');
  const kws   = (domainInfo.keywords ?? []).map(k => `#${k}`).join(' ');
  const now   = new Date().toISOString();
  const mapMd = [
    '---',
    `domain: ${domainName}`,
    `description: ${domainInfo.description ?? ''}`,
    `created: ${now}`,
    `auto_detected: true`,
    '---',
    '',
    `# ${domainName} · 维度图谱`,
    '',
    `> ${domainInfo.description ?? ''}`,
    '',
    '## 核心维度',
    dims,
    '',
    '## 关键词',
    kws,
    '',
    '## 层级结构',
    `- [[L0]] — 原始信息、未加工片段`,
    `- [[L1/${category}]] — 经过整理的知识点`,
    `- [[L2/${category}]] — 跨域关联洞见`,
    `- [[L3/${category}]] — 提炼的高阶原则`,
  ].join('\n');

  await writeFile(join(domainDir, '_维度图谱.md'), mapMd, 'utf8');
  return `L1/${category}/_维度图谱.md`;
}

async function runDomainDetectAgent(logger) {
  // 1. Scroll all domain=null active records with vectors
  const unassigned = [];
  let offset = null;
  do {
    const body = {
      limit: 500,
      with_payload: true,
      with_vector: true,
      filter: {
        should: [
          { is_null: { key: 'domain' } },
          { key: 'domain', match: { value: '未分类' } },
        ],
        must_not: [
          { key: 'status', match: { value: 'superseded' } },
          { key: 'source', match: { value: 'distill' } },  // distill L3 不参与域聚类，应通过 _storeDistilled 直接继承域
        ],
      },
    };
    if (offset != null) body.offset = offset;
    const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', body);
    if (!r.ok) break;
    const pts = r.body?.result?.points ?? [];
    unassigned.push(...pts);
    offset = r.body?.result?.next_page_offset ?? null;
  } while (offset != null);

  if (unassigned.length < CLUSTER_MIN_SIZE) {
    logger?.info?.(`[atlas-memory] 域检测Agent: 待分类记录${unassigned.length}条（含domain=null和'未分类'），未达聚类最低${CLUSTER_MIN_SIZE}条，跳过`);
    return { checked: unassigned.length, clusters_found: 0, new_domains: 0, assigned: 0 };
  }

  logger?.info?.(`[atlas-memory] 域检测Agent: 扫描${unassigned.length}条未分类记录`);

  // 2. Cluster
  const clusters = clusterNodes(unassigned.filter(n => n.vector));
  logger?.info?.(`[atlas-memory] 域检测Agent: 聚类${clusters.length}个`);

  let newDomains = 0;
  let assigned = 0;

  for (const cluster of clusters) {
    const vectors = cluster.map(n => n.vector);
    const c = centroid(vectors);

    // 3. Compare centroid against existing domains
    const match = await matchDomainForVector(c);

    let targetDomain;
    if (match.domain && match.score >= DOMAIN_MATCH_SCORE) {
      // Assign to existing domain
      targetDomain = match.domain;
    } else {
      // 新域至少需要2条L1知识记录，避免单条 distill/L3 数据创建空域
      const l1Count = cluster.filter(n => (n.payload?.level ?? 0) === LEVEL_KNOWLEDGE).length;
      if (l1Count < 2) {
        logger?.info?.(`[atlas-memory] 域检测Agent: 跳过聚类（L1=${l1Count}<2），不创建新域`);
        continue;
      }
      // 4. Infer new domain via DeepSeek
      const domainInfo = await inferNewDomain(cluster);
      if (!domainInfo) continue;

      const newName = domainInfo.domain_name;
      if (DOMAIN_DIRS[newName]) {
        // Race condition: domain was just created; just assign
        targetDomain = newName;
      } else {
        // Create directory + map
        await createDomainStructure(newName, domainInfo);

        // 5. Update runtime caches
        DOMAIN_DIRS[newName] = newName;
        DOMAIN_DESCRIPTIONS[newName] = domainInfo.description;
        const vec = await embed(domainInfo.description);
        if (vec) domainEmbeddingCache.set(newName, vec);

        targetDomain = newName;
        newDomains++;

        appendEvolutionLog('DOMAIN_NEW',
          `新域: "${newName}" — ${domainInfo.description} (从${cluster.length}条记录聚类发现)`
        ).catch(() => {});
        logger?.info?.(`[atlas-memory] 域检测Agent: 新域 "${newName}"`);
      }
    }

    // 6. Patch all nodes in cluster → target domain (batch via raw HTTP)
    const now = new Date().toISOString();
    await writeQueue.push(WRITE_PRIORITY.AGENT, async () => {
      const ids = cluster.map(n => n.id);
      await httpReq(
        `${QDRANT}/collections/${COLLECTION}/points/payload`, 'POST',
        { payload: { domain: targetDomain, last_verified: now }, points: ids },
      );
    });

    // Also move Obsidian file if obsidian_path exists and is in _未分类
    for (const node of cluster) {
      const op = node.payload?.obsidian_path;
      if (op && op.startsWith('L0/') && OBSIDIAN_VAULT) {
        const src = join(OBSIDIAN_VAULT, op);
        try {
          const content = await readFile(src, 'utf8').catch(() => null);
          if (content) {
            const updated = content.replace(/^domain: 未分类$/m, `domain: ${targetDomain}`);
            await writeFile(src, updated, 'utf8');
          }
        } catch {}
      }
    }

    assigned += cluster.length;
    appendEvolutionLog('DOMAIN_ASSIGN',
      `域归属: "${targetDomain}" ← ${cluster.length}条 (相似度${match.score?.toFixed(2) ?? 'new'})`
    ).catch(() => {});
  }

  logger?.info?.(`[atlas-memory] 域检测Agent: 新域${newDomains}个, 归属${assigned}条`);
  return { checked: unassigned.length, clusters_found: clusters.length, new_domains: newDomains, assigned };
}

// ── Phase 5：关联Agent ────────────────────────────────────────────────────────

let lastAssociateRun = 0; // unix ms，用于只取上轮以来新增的L1

async function generateCrossInsight(nodeA, nodeB) {
  const da = nodeA.payload?.domain ?? '未分类';
  const db = nodeB.payload?.domain ?? '未分类';
  const sameDomain = da === db;

  // 使用完整内容，不只是 summary
  const fullA = [
    nodeA.payload?.summary ?? '',
    (nodeA.payload?.key_points ?? []).join('；'),
    nodeA.payload?.reasoning ?? '',
  ].filter(Boolean).join('\n').slice(0, 800);
  const fullB = [
    nodeB.payload?.summary ?? '',
    (nodeB.payload?.key_points ?? []).join('；'),
    nodeB.payload?.reasoning ?? '',
  ].filter(Boolean).join('\n').slice(0, 800);

  const topicA = nodeA.payload?.topic ?? '知识A';
  const topicB = nodeB.payload?.topic ?? '知识B';

  const sys = '你是深度知识关联专家。严格只输出有效JSON对象，不要任何解释或markdown代码块。只基于提供的两段知识内容进行关联分析，禁止引入外部事实、数据或假设。';
  const user =
    `深度${sameDomain ? '同域' : '跨域'}关联分析：\n\n` +
    `【${topicA}】（${da}）\n${fullA}\n\n` +
    `【${topicB}】（${db}）\n${fullB}\n\n` +
    `请进行深度关联分析，输出JSON：\n` +
    `{\n` +
    `  "association_type": "causal|complement|contrast|hierarchy|sequence|amplify|prerequisite|paradox",\n` +
    `  "mechanism": "这两条知识在底层结构/逻辑上为什么能产生关联？（30-60字）",\n` +
    `  "insight": "这个关联揭示了什么更深层的规律或原则？（60-100字，说明启示而非只描述关联）",\n` +
    `  "decision_implication": "这个关联对实际决策/行动有什么具体指导？（30-50字，可操作结论）",\n` +
    `  "application_protocol": ["如何使用这个关联的步骤1（15-25字）", "步骤2", "步骤3"],\n` +
    `  "predictive_power": "当A发生时B会出现什么变化？能预测什么结果（20-40字）",\n` +
    `  "domain_transfer": "${sameDomain ? '' : '将' + da + '域原则迁移到' + db + '域的具体操作方式（20-40字）'}",\n` +
    `  "conditions": "关联成立的条件与不成立的条件（20-40字）",\n` +
    `  "synthesis_potential": "是否暗示存在更大框架？（15-30字，无则填空字符串）",\n` +
    `  "is_contradiction": false,\n` +
    `  "insight_depth": 0.8,\n` +
    `  "source_grounding": "分别引用A和B中支撑此洞见的原句或核心词（20-40字，若无法从原文找到依据则填""）"\n` +
    `}`;

  const raw = await deepseekGenerate(sys, user, 600);
  if (!raw) {
    // fallback to omlx with simpler prompt
    const simpleUser =
      `${sameDomain ? '同域' : '跨域'}知识关联分析：\n` +
      `知识A（${da}）：${fullA.slice(0,300)}\n\n知识B（${db}）：${fullB.slice(0,300)}\n\n` +
      `输出JSON：{"association_type":"causal|complement|contrast|hierarchy|sequence","mechanism":"底层机制（20-40字）",` +
      `"insight":"深层规律（50-80字）","decision_implication":"决策启示（20-30字）","conditions":"适用条件（10-20字）",` +
      `"synthesis_potential":"","is_contradiction":false,"insight_depth":0.6}`;
    const r2 = await omlxGenerate(sys, simpleUser, 400, AGENT_OMLX_TIMEOUT_MS);
    if (!r2) return null;
    const p2 = (() => { try { return JSON.parse(r2.replace(/```[a-z]*\n?/gi,'').replace(/```/g,'').trim()); } catch { return null; } })();
    if (!p2?.insight || p2.insight.length < 30) return null;
    return p2;
  }

  const parsed = (() => { try { return JSON.parse(raw.replace(/```[a-z]*\n?/gi,'').replace(/```/g,'').trim()); } catch { return null; } })();
  if (!parsed?.insight || parsed.insight.length < 30) return null;
  // 无来源依据说明时降低深度评分（防止幻觉洞见被高分保留）
  if (!parsed.source_grounding || parsed.source_grounding.length < 10) {
    parsed.insight_depth = Math.max(0, (parsed.insight_depth ?? 0.7) - 0.2);
  }
  // 过滤低深度关联（不值得存储）
  if ((parsed.insight_depth ?? 1.0) < 0.4) return null;
  return parsed;
}

async function writeL2Obsidian(domain, topic, insightObj, srcAPath, srcBPath, domainB, srcALabel, srcBLabel) {
  if (!OBSIDIAN_VAULT) return null;
  const domainDir = await getCategoryForDomain(domain);
  const dir = join(OBSIDIAN_VAULT, 'L2', domainDir);
  await mkdir(dir, { recursive: true });
  const assocType      = insightObj.association_type ?? 'complement';
  const insightText    = typeof insightObj === 'string' ? insightObj : (insightObj.insight ?? '');
  const isContradiction = insightObj.is_contradiction ?? false;
  const typeLabel      = { causal:'因果', complement:'互补', contrast:'对立', hierarchy:'层级', sequence:'时序', amplify:'放大', prerequisite:'前提', paradox:'悖论' }[assocType] ?? assocType;
  const slug = topic.replace(/[/\\:*?"<>|]/g, '-').slice(0, 40);
  const date = new Date().toISOString().slice(0, 10);
  const filename = `${slug}-${date}.md`;
  const linkA = srcAPath ? `[[${srcAPath.replace(/\.md$/, '')}]]` : '';
  const linkB = srcBPath ? `[[${srcBPath.replace(/\.md$/, '')}]]` : '';
  const protocol = (insightObj.application_protocol ?? []).map((s, i) => `${i+1}. ${s}`).join('\n');
  const md = [
    '---',
    `level: L2`,
    `domain: ${domain}`,
    `linked_domain: ${domainB}`,
    `topic: ${topic}`,
    `association_type: ${assocType}`,
    `insight_depth: ${insightObj.insight_depth ?? 0}`,
    `is_contradiction: ${isContradiction}`,
    `created: ${new Date().toISOString()}`,
    '---',
    '',
    `# ${topic}`,
    `> 关联类型：${typeLabel}  深度：${((insightObj.insight_depth ?? 0) * 100).toFixed(0)}%${isContradiction ? '  ⚠️ 争议点' : ''}`,
    '',
    '## 深层洞见',
    insightText,
    '',
    '## 底层机制',
    insightObj.mechanism ?? '',
    '',
    '## 决策指导',
    insightObj.decision_implication ?? '',
    '',
    ...(protocol ? ['## 使用协议', protocol, ''] : []),
    ...(insightObj.predictive_power ? ['## 预测力', insightObj.predictive_power, ''] : []),
    ...(!domainB || domainB === domain || !insightObj.domain_transfer ? [] : ['## 跨域迁移', insightObj.domain_transfer, '']),
    '## 成立条件',
    insightObj.conditions ?? '',
    '',
    '## 来源',
    `- ${linkA || srcAPath || srcALabel || '(未知)'}`,
    `- ${linkB || srcBPath || srcBLabel || '(未知)'}`,
    ...(insightObj.source_grounding ? ['', '## 原文依据', insightObj.source_grounding] : []),
  ].join('\n');
  await writeFile(join(dir, filename), md, 'utf8');
  return `L2/${domainDir}/${filename}`;
}

async function appendWikilink(obsidianPath, linkTarget) {
  if (!OBSIDIAN_VAULT || !obsidianPath) return;
  const fullPath = join(OBSIDIAN_VAULT, obsidianPath);
  // 幂等：已含该链接则跳过
  const existing = await readFile(fullPath, 'utf8').catch(() => '');
  const linkStr = `[[${linkTarget.replace(/\.md$/, '')}]]`;
  if (existing.includes(linkStr)) return;
  await appendFile(fullPath, `\n- ${linkStr}\n`, 'utf8').catch(() => {});
}

async function qdrantSearchL1(vector, excludeId) {
  // 直接走 httpReq，支持 level + status 过滤 + 排除自身
  const body = {
    vector,
    limit: 10,
    with_payload: true,
    score_threshold: ASSOC_MIN_SCORE,
    filter: {
      must: [
        { key: 'level',  match: { value: LEVEL_KNOWLEDGE } },
        { key: 'status', match: { value: 'active' } },
      ],
      must_not: [
        { key: 'status', match: { value: 'superseded' } },
        { has_id: [excludeId] },
      ],
    },
  };
  const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/search`, 'POST', body);
  return r.ok ? (r.body?.result ?? []) : [];
}

async function qdrantSearchSameDomain(vector, excludeId, domain) {
  const body = {
    vector,
    limit: 5,
    with_payload: true,
    score_threshold: 0.86,
    filter: {
      must: [
        { key: 'level',  match: { value: LEVEL_KNOWLEDGE } },
        { key: 'status', match: { value: 'active' } },
        { key: 'domain', match: { value: domain } },
      ],
      must_not: [
        { key: 'status', match: { value: 'superseded' } },
        { has_id: [excludeId] },
      ],
    },
  };
  const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/search`, 'POST', body);
  return r.ok ? (r.body?.result ?? []) : [];
}

async function runAssociateAgent(logger) {
  const since = lastAssociateRun;
  lastAssociateRun = Date.now();
  // 持久化到磁盘，防止重启后重复处理全量 L1
  writeFile(STATE_FILE, JSON.stringify({ lastAssociateRun }), 'utf8').catch(() => {});

  // 预加载已有 L2 关联对，防止跨轮重复生成
  const existingL2Pairs = new Set();
  try {
    let l2Off = null;
    do {
      const lr = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', {
        limit: 500, with_payload: ['associated_ids'], with_vector: false,
        filter: { must: [{ key: 'level', match: { value: LEVEL_INSIGHT } }], must_not: [{ match: { key: 'status', value: 'superseded' } }] },
        ...(l2Off != null ? { offset: l2Off } : {}),
      });
      if (!lr.ok) break;
      for (const pt of lr.body?.result?.points ?? []) {
        const ids = pt.payload?.associated_ids ?? [];
        if (ids.length >= 2) existingL2Pairs.add(ids.map(String).sort().join(':'));
      }
      l2Off = lr.body?.result?.next_page_offset ?? null;
    } while (l2Off != null);
  } catch {}
  logger?.info?.(`[atlas-memory] 关联Agent: 已有L2对${existingL2Pairs.size}个`);


  // 1. Fetch L1 nodes added since last run (or all L1 if first run)
  const l1Nodes = [];
  let offset = null;
  // 使用 last_verified 而非 created_at，因为 L0→L1 晋升时 last_verified 被更新为当前时间，
  // 而 created_at 是 L0 原始入库时间，早于 lastAssociateRun 会导致新节点被漏掉
  const scrollFilter = since > 0
    ? { must: [
        { key: 'level',  match: { value: LEVEL_KNOWLEDGE } },
        { key: 'status', match: { value: 'active' } },
        { key: 'last_verified', range: { gte: new Date(since).toISOString() } },
      ] }
    : { must: [
        { key: 'level',  match: { value: LEVEL_KNOWLEDGE } },
        { key: 'status', match: { value: 'active' } },
      ] };

  do {
    const body = { limit: 200, with_payload: true, with_vector: true, filter: scrollFilter };
    if (offset != null) body.offset = offset;
    const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', body);
    if (!r.ok) break;
    l1Nodes.push(...(r.body?.result?.points ?? []));
    offset = r.body?.result?.next_page_offset ?? null;
  } while (offset != null);

  if (!l1Nodes.length) {
    logger?.info?.('[atlas-memory] 关联Agent: 无新L1节点，跳过');
    return { checked: 0, created: 0 };
  }

  logger?.info?.(`[atlas-memory] 关联Agent: 检查${l1Nodes.length}个L1节点`);
  let created = 0;
  // 本轮已处理对，防止 A→B 和 B→A 重复生成
  const processedPairs = new Set();

  for (const node of l1Nodes) {
    const nodeDomain = node.payload?.domain ?? null;

    // 2. Cross-domain similarity search，仅返回 L1 active 节点
    const hits = await qdrantSearchL1(node.vector, node.id);

    // 过滤：不同域 + 分数在 (ASSOC_MIN_SCORE, ASSOC_MAX_SCORE]
    const crossCandidates = hits.filter(h => {
      const hd = h.payload?.domain ?? null;
      return h.score <= ASSOC_MAX_SCORE && hd !== nodeDomain;
    });

    // 同域深度关联（score > 0.86，同域内印证/矛盾）
    const sameDomainCandidates = nodeDomain
      ? await qdrantSearchSameDomain(node.vector, node.id, nodeDomain)
      : [];

    const allCandidates = [...crossCandidates.slice(0, 5), ...sameDomainCandidates.slice(0, 2)];

    for (const partner of allCandidates) {
      // 去重：当前轮 + 跨轮（existingL2Pairs）
      const pairKey = [node.id, partner.id].map(String).sort().join(':');
      if (processedPairs.has(pairKey) || existingL2Pairs.has(pairKey)) continue;
      processedPairs.add(pairKey);
      existingL2Pairs.add(pairKey); // 本轮新建的也加入，防止同轮两次生成

      const partnerDomain = partner.payload?.domain ?? null;
      const isSameDomain  = partnerDomain === nodeDomain;

      // 3. Generate insight with association type via omlx
      const insightObj = await generateCrossInsight(node, partner);
      if (!insightObj?.insight) continue;

      const topicA = node.payload?.topic ?? node.payload?.tags?.[0] ?? '知识';
      const topicB = partner.payload?.topic ?? partner.payload?.tags?.[0] ?? '知识';
      const insightTopic = `${topicA}×${topicB}`;
      const assocType    = insightObj.association_type ?? 'complement';

      const pathA = node.payload?.obsidian_path ?? null;
      const pathB = partner.payload?.obsidian_path ?? null;
      const domainADir = nodeDomain ?? '未分类';
      const domainBDir = partnerDomain ?? '未分类';

      // 4. 两个域各写一份 L2 文件（同域只写一份）
      let l2PathA = null, l2PathB = null;
      try { l2PathA = await writeL2Obsidian(domainADir, insightTopic, insightObj, pathA, pathB, domainBDir, topicA, topicB); } catch {}
      if (!isSameDomain) {
        try { l2PathB = await writeL2Obsidian(domainBDir, insightTopic, insightObj, pathB, pathA, domainADir, topicB, topicA); } catch {}
      }

      // 5. 各自 wikilink 指向自己域的 L2 文件（幂等写入）
      if (l2PathA) try { await appendWikilink(pathA, l2PathA); } catch {}
      if (l2PathB) try { await appendWikilink(pathB, l2PathB); } catch {}
      else if (l2PathA) try { await appendWikilink(pathB, l2PathA); } catch {}

      // 6. Qdrant：L2 节点（存储完整深度字段）
      const insightContent = [
        insightObj.insight,
        insightObj.mechanism ? `机制：${insightObj.mechanism}` : '',
        insightObj.decision_implication ? `决策启示：${insightObj.decision_implication}` : '',
      ].filter(Boolean).join('\n');
      const vector = await embed(insightContent);
      if (vector) {
        const now = new Date().toISOString();
        const domainTag = isSameDomain ? 'same-domain' : 'cross-domain';
        const base = {
          content: insightContent, category: 'work',
          importance: insightObj.is_contradiction ? 'high' : 'medium',
          tags: [domainADir, domainBDir, domainTag, assocType],
          memory_type: 'insight', created_at: now,
          source: 'associate-agent', session_key: 'agent',
          hit_count: 0, last_accessed_at: null, status: 'active',
          feedback_score: 1.0, level: LEVEL_INSIGHT,
          topic: insightTopic, freshness_score: 1.0, decay_rate: 'medium',
          last_verified: now, source_ids: [node.id, partner.id],
          associated_ids: [node.id, partner.id],
          derived_to_id: null, acquisition_source: 'associate-agent',
          association_type: assocType,
          is_contradiction: insightObj.is_contradiction ?? false,
          mechanism:            insightObj.mechanism            ?? '',
          decision_implication: insightObj.decision_implication ?? '',
          conditions:           insightObj.conditions           ?? '',
          synthesis_potential:  insightObj.synthesis_potential  ?? '',
          insight_depth:        insightObj.insight_depth        ?? 0.7,
          linked_domain:        domainBDir,
          source_grounding:     insightObj.source_grounding     ?? '',
        };
        await writeQueue.push(WRITE_PRIORITY.AGENT, () =>
          upsert(vector, { ...base, domain: domainADir, obsidian_path: l2PathA })
        );
        if (l2PathB) {
          await writeQueue.push(WRITE_PRIORITY.AGENT, () =>
            upsert(vector, { ...base, domain: domainBDir, linked_domain: domainADir, obsidian_path: l2PathB })
          );
        }
        created++;
        appendEvolutionLog('ASSOCIATE',
          `L2洞见[${assocType}]: "${insightTopic}" [${domainADir}×${domainBDir}]${insightObj.is_contradiction ? ' ⚠️争议' : ''}`
        ).catch(() => {});
      }
    }
  }

  logger?.info?.(`[atlas-memory] 关联Agent: 新建${created}个L2洞见`);
  return { checked: l1Nodes.length, created };
}

// ── Phase 6：合成Agent ────────────────────────────────────────────────────────

async function getNextL3Version(domain) {
  // Scan existing L3 nodes for this domain, return max version + 1
  const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', {
    limit: 100,
    with_payload: true,
    with_vector: false,
    filter: {
      must: [
        { key: 'level',  match: { value: LEVEL_WISDOM } },
        { key: 'domain', match: { value: domain } },
        { key: 'status', match: { value: 'active' } },
      ],
    },
  });
  const pts = r.body?.result?.points ?? [];
  let max = 0;
  for (const pt of pts) {
    const m = (pt.payload?.topic ?? '').match(/v(\d+)$/);
    if (m) max = Math.max(max, parseInt(m[1], 10));
  }
  return max + 1;
}

async function synthesizeL3(cluster, domain, upgradeNotes = '') {
  // 使用完整的 L2 内容（包括 mechanism、decision_implication 等深度字段）
  const excerpts = cluster
    .map((n, i) => {
      const p = n.payload ?? {};
      const assocType = p.association_type ? `[${p.association_type}]` : '';
      const parts = [
        `洞见${i + 1}${assocType}：${p.content ?? ''}`.slice(0, 500),
        p.mechanism           ? `  机制：${p.mechanism}` : '',
        p.decision_implication ? `  决策：${p.decision_implication}` : '',
        p.conditions           ? `  条件：${p.conditions}` : '',
      ].filter(Boolean).join('\n');
      return parts;
    })
    .join('\n\n');

  const sys = '你是战略框架合成专家。严格输出JSON，不要markdown代码块，不要解释。只综合以下L2洞见中明确存在的内容，禁止发明洞见未包含的事实或数据。';
  const upgradeCtx = upgradeNotes ? `\n\n本次升级背景：${upgradeNotes}` : '';
  const user =
    `以下是来自"${domain}"领域的深度知识洞见集群（共${cluster.length}条）：\n\n${excerpts}${upgradeCtx}\n\n` +
    `请综合这些洞见，生成一套可直接指导行动的框架。输出JSON：\n` +
    `{\n` +
    `  "title": "框架名称（不超过10字）",\n` +
    `  "summary": "核心论点（40-80字，说明框架解决什么问题、核心主张是什么）",\n` +
    `  "reasoning_chain": "推理逻辑：这些洞见如何推导出此框架？关键推理步骤（60-100字）",\n` +
    `  "principles": [{"rule":"原则表述（15-25字）","because":"为什么有效，底层逻辑（15-25字）","trigger":"何时激活：出现什么信号时使用（10-20字）"}],\n` +
    `  "actions": [{"action":"具体行动（15-25字）","when":"触发条件（10-20字）","success_signal":"成功的标志是什么（10-20字）"}],\n` +
    `  "failure_modes": ["最常见失败方式及原因（20-30字）","第二种失败方式"],\n` +
    `  "prerequisites": "使用此框架需要的前提条件（20-35字）",\n` +
    `  "conditions": "适用场景（20-40字）",\n` +
    `  "counter_evidence": "该框架最强的反驳是什么？已知反例或局限（40-60字）",\n` +
    `  "evidence_base": "框架依赖的核心证据（30-50字）",\n` +
    `  "knowledge_gaps": ["若要验证此框架，最需要补充的知识问题1","问题2","问题3"],\n` +
    `  "meta_pattern": "该框架是否折射出更大的跨域规律？（15-30字，无则填空字符串）",\n` +
    `  "source_reliability": 1.0\n` +
    `}\n（source_reliability: 0.0~1.0，1.0=框架所有内容均来自上述洞见，0.5=有额外推断）`;

  const raw = (await deepseekGenerate(sys, user, 1200)) ?? (await omlxGenerate(sys, user, 1000, AGENT_OMLX_TIMEOUT_MS));
  if (!raw) return null;
  try {
    const cleaned = raw.replace(/```[a-z]*\n?/gi, '').replace(/```/g, '').trim();
    const parsed = JSON.parse(cleaned);
    if (!parsed.title || !parsed.summary) return null;
    // 兼容旧格式（principles 可能是字符串数组）
    parsed.principles = Array.isArray(parsed.principles)
      ? parsed.principles.slice(0, 5).map(p => typeof p === 'string' ? { rule: p, because: '', trigger: '' } : p)
      : [];
    parsed.actions = Array.isArray(parsed.actions)
      ? parsed.actions.slice(0, 5).map(a => typeof a === 'string' ? { action: a, when: '', success_signal: '' } : a)
      : [];
    parsed.failure_modes    = Array.isArray(parsed.failure_modes)    ? parsed.failure_modes.slice(0, 3)    : [];
    parsed.knowledge_gaps   = Array.isArray(parsed.knowledge_gaps)   ? parsed.knowledge_gaps.slice(0, 3)   : [];
    parsed.reasoning_chain  = parsed.reasoning_chain  ?? '';
    parsed.counter_evidence = parsed.counter_evidence ?? '';
    parsed.evidence_base    = parsed.evidence_base    ?? '';
    parsed.prerequisites    = parsed.prerequisites    ?? '';
    parsed.meta_pattern     = parsed.meta_pattern     ?? '';
    parsed.source_reliability = typeof parsed.source_reliability === 'number'
      ? parseFloat(parsed.source_reliability.toFixed(2)) : 1.0;
    parsed.hallucination_risk = parsed.source_reliability < 0.6;

    // 自我挑战轮：找最弱点，且必须输出 revised_principle 形成闭环
    const principleText = parsed.principles.map(p => p.rule ?? p).join('；');
    if (principleText) {
      const challengeSys = '你是严格的批判性思维专家。输出JSON，不要解释。';
      const challengeUser =
        `对以下框架进行最强力的批判性审查：\n` +
        `框架：${parsed.title}\n核心主张：${parsed.summary}\n原则：${principleText}\n` +
        `输出JSON：{\n` +
        `  "weakest_assumption": "框架最脆弱的假设（20-40字）",\n` +
        `  "attack_scenario": "最可能推翻此框架的场景（20-40字）",\n` +
        `  "missing_variable": "框架忽视的最重要变量（15-30字）",\n` +
        `  "revised_principle": "基于以上批判，框架最需要修订的一条原则的新表述（20-35字）"\n` +
        `}`;
      const cr = await deepseekGenerate(challengeSys, challengeUser, 400);
      if (cr) {
        try {
          parsed.challenge = JSON.parse(cr.replace(/```[a-z]*\n?/gi,'').replace(/```/g,'').trim());
        } catch {}
      }
    }

    return parsed;
  } catch { return null; }
}

// L4 元合成：从多个 L3 框架中提炼跨域元规律
async function synthesizeL4(l3Nodes) {
  if (l3Nodes.length < 3) return null;
  const frameExcerpts = l3Nodes.map((n, i) => {
    const p = n.payload ?? {};
    return `框架${i+1}【${p.domain}·${p.topic}】\n核心：${p.content?.slice(0,200) ?? ''}\n` +
           (p.meta_pattern ? `元模式提示：${p.meta_pattern}` : '');
  }).join('\n\n');

  const sys = '你是元认知合成专家。严格输出JSON，不要解释。';
  const user =
    `以下是来自不同领域的${l3Nodes.length}个知识框架，请发现它们共同折射的跨域元规律：\n\n${frameExcerpts}\n\n` +
    `输出JSON：\n` +
    `{\n` +
    `  "meta_title": "元规律名称（不超过12字）",\n` +
    `  "meta_principle": "跨域通用原则（50-80字，这些框架共同指向什么更深层的人类规律？）",\n` +
    `  "domain_manifestations": [{"domain":"域名","manifestation":"该规律在此域的具体体现（15-25字）"}],\n` +
    `  "universal_action": "基于此元规律，最普适的行动指南（20-40字）",\n` +
    `  "recursive_question": "若此规律为真，它对自身最强的挑战是什么？（20-35字）"\n` +
    `}`;

  const raw = await deepseekGenerate(sys, user, 800);
  if (!raw) return null;
  try {
    const cleaned = raw.replace(/```[a-z]*\n?/gi,'').replace(/```/g,'').trim();
    const parsed = JSON.parse(cleaned);
    if (!parsed.meta_title || !parsed.meta_principle) return null;
    parsed.domain_manifestations = Array.isArray(parsed.domain_manifestations) ? parsed.domain_manifestations : [];
    return parsed;
  } catch { return null; }
}

async function writeL3Obsidian(domain, framework, version, sourceIds) {
  if (!OBSIDIAN_VAULT) return null;
  const domainDir = await getCategoryForDomain(domain);
  const dir = join(OBSIDIAN_VAULT, 'L3', domainDir);
  await mkdir(dir, { recursive: true });
  const filename = `${domain}打法-v${version}.md`;
  const principles = (framework.principles ?? []).map(p =>
    typeof p === 'string' ? `- ${p}` : `- **${p.rule ?? ''}** — ${p.because ?? ''}${p.trigger ? ` (触发：${p.trigger})` : ''}`
  ).join('\n');
  const actions    = (framework.actions ?? []).map(a =>
    typeof a === 'string' ? `- [ ] ${a}` : `- [ ] ${a.action ?? ''} | 触发：${a.when ?? ''} → 成功标志：${a.success_signal ?? ''}`
  ).join('\n');
  const sourceLinks = sourceIds.map(id => `- ID:${id}`).join('\n');
  const now = new Date().toISOString();
  const md = [
    '---',
    `level: L3`,
    `domain: ${domain}`,
    `version: v${version}`,
    `title: ${framework.title}`,
    `created: ${now}`,
    `source_count: ${sourceIds.length}`,
    '---',
    '',
    `# ${framework.title}`,
    `> v${version} · ${domain} · ${now.slice(0, 10)}`,
    '',
    `## 核心论点`,
    framework.summary,
    '',
    `## 推理链条`,
    framework.reasoning_chain ?? '',
    '',
    `## 原则`,
    principles,
    '',
    `## 行动清单`,
    actions,
    '',
    `## 适用条件`,
    framework.conditions ?? '',
    '',
    `## 证据基础`,
    framework.evidence_base ?? '',
    '',
    `## 已知局限/反例`,
    framework.counter_evidence ?? '',
    '',
    `## 来源洞见`,
    sourceLinks,
  ].join('\n');
  await writeFile(join(dir, filename), md, 'utf8');
  return `L3/${domainDir}/${filename}`;
}

async function writeL4Obsidian(meta, sourceL3Count) {
  if (!OBSIDIAN_VAULT) return null;
  const dir = join(OBSIDIAN_VAULT, 'L4');
  await mkdir(dir, { recursive: true });
  const safeTitle = (meta.meta_title ?? 'meta').replace(/[/\\:*?"<>|]/g, '-').slice(0, 60);
  const filename = `${safeTitle}-L4.md`;
  const now = new Date().toISOString();
  const manifestations = (meta.domain_manifestations ?? []).map(m =>
    `- **${m.domain}**: ${m.manifestation}`
  ).join('\n');
  const md = [
    '---',
    `level: L4`,
    `domain: META`,
    `title: ${meta.meta_title}`,
    `created: ${now}`,
    `source_l3_count: ${sourceL3Count}`,
    '---',
    '',
    `# ${meta.meta_title}`,
    `> L4 元规律 · META · ${now.slice(0, 10)}`,
    '',
    `## 跨域元原则`,
    meta.meta_principle ?? '',
    '',
    `## 各域体现`,
    manifestations,
    '',
    `## 通用行动指南`,
    meta.universal_action ?? '',
    '',
    `## 递归自反挑战`,
    meta.recursive_question ?? '',
  ].join('\n');
  await writeFile(join(dir, filename), md, 'utf8');
  return `L4/${filename}`;
}

async function runSynthesizeAgent(logger) {
  // 1. Fetch all L2 active nodes with vectors
  const l2Nodes = [];
  let offset = null;
  do {
    const body = {
      limit: 500, with_payload: true, with_vector: true,
      filter: {
        must: [
          { key: 'level',  match: { value: LEVEL_INSIGHT } },
          { key: 'status', match: { value: 'active' } },
        ],
      },
    };
    if (offset != null) body.offset = offset;
    const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', body);
    if (!r.ok) break;
    l2Nodes.push(...(r.body?.result?.points ?? []));
    offset = r.body?.result?.next_page_offset ?? null;
  } while (offset != null);

  if (!l2Nodes.length) {
    logger?.info?.('[atlas-memory] 合成Agent: 无L2节点，跳过');
    return { l2_scanned: 0, clusters: 0, synthesized: 0 };
  }

  logger?.info?.(`[atlas-memory] 合成Agent: 扫描${l2Nodes.length}个L2节点`);

  // 2. Cluster L2 nodes by domain then by vector similarity
  const byDomain = new Map();
  for (const n of l2Nodes) {
    const d = n.payload?.domain ?? '未分类';
    if (!byDomain.has(d)) byDomain.set(d, []);
    byDomain.get(d).push(n);
  }

  let synthesized = 0;
  let totalClusters = 0;

  for (const [domain, nodes] of byDomain) {
    // Skip nodes already derived to an L3
    const underivedNodes = nodes.filter(n => !n.payload?.derived_to_id);
    if (underivedNodes.length < CLUSTER_MIN_SIZE) continue;

    const clusters = clusterNodes(underivedNodes, CLUSTER_MIN_SCORE, CLUSTER_MIN_SIZE);
    totalClusters += clusters.length;

    for (const cluster of clusters) {
      // 检查是否已有同域L3：有则记录升级背景
      const existingVersion = await getNextL3Version(domain);
      const isUpdate = existingVersion > 1;
      const upgradeNotes = isUpdate
        ? `基于${cluster.length}条新洞见（含${cluster.filter(n => n.payload?.is_contradiction).length}条争议点）更新框架`
        : '';
      // Collect existing active L3 IDs for this domain (to supersede after new version is created)
      let prevL3Ids = [];
      if (isUpdate) {
        const prevR = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', {
          limit: 50, with_payload: false, with_vector: false,
          filter: { must: [{ key: 'level', match: { value: LEVEL_WISDOM } }, { key: 'domain', match: { value: domain } }, { key: 'status', match: { value: 'active' } }] },
        });
        prevL3Ids = (prevR.body?.result?.points ?? []).map(p => p.id);
      }

      // 3. DeepSeek synthesis
      const framework = await synthesizeL3(cluster, domain, upgradeNotes);
      if (!framework) continue;

      // 4. Version numbering
      const version = existingVersion;
      const topic   = `${framework.title}-v${version}`;

      // 5. Write L3 Obsidian
      const sourceIds = cluster.map(n => n.id);
      const l3Path = await writeL3Obsidian(domain, framework, version, sourceIds);

      // 6. Upsert L3 Qdrant node
      const principlesText = (framework.principles ?? []).map(p => typeof p === 'string' ? p : (p.rule ?? '')).join('；');
      const content = `${framework.title}：${framework.summary}。原则：${principlesText}`;
      const vector  = await embed(content);
      if (!vector) continue;

      let l3Id;
      await writeQueue.push(WRITE_PRIORITY.AGENT, async () => {
        const now = new Date().toISOString();
        const res = await upsert(vector, {
          content, category: 'work', importance: 'critical',
          title: framework.title ?? topic,
          tags: [domain, 'framework', `v${version}`],
          memory_type: 'distilled', created_at: now,
          source: 'synthesize-agent', session_key: 'agent',
          hit_count: 0, last_accessed_at: null, status: 'active',
          feedback_score: 1.0, level: LEVEL_WISDOM,
          domain, topic, freshness_score: 1.0, decay_rate: 'slow',
          last_verified: now, source_ids: sourceIds,
          associated_ids: [], derived_to_id: null,
          obsidian_path: l3Path, acquisition_source: 'synthesize-agent',
          reasoning_chain:  framework.reasoning_chain  ?? '',
          counter_evidence: framework.counter_evidence ?? '',
          evidence_base:    framework.evidence_base    ?? '',
          knowledge_gaps:      framework.knowledge_gaps      ?? [],
          meta_pattern:        framework.meta_pattern        ?? '',
          challenge:           framework.challenge           ?? null,
          failure_modes:       framework.failure_modes       ?? [],
          prerequisites:       framework.prerequisites       ?? '',
          source_reliability:  framework.source_reliability  ?? 1.0,
          hallucination_risk:  framework.hallucination_risk  ?? false,
        });
        l3Id = res?.id;
      });

      // 7. Patch source L2 nodes: derived_to_id → l3Id
      if (l3Id) {
        await writeQueue.push(WRITE_PRIORITY.AGENT, () =>
          httpReq(`${QDRANT}/collections/${COLLECTION}/points/payload`, 'POST',
            { payload: { derived_to_id: l3Id }, points: sourceIds })
        );
      } else {
        logger?.warn?.(`[atlas-memory] 合成Agent: L3存储失败，无法更新L2 derived_to_id`);
      }

      // Supersede previous L3 versions for this domain
      if (l3Id && prevL3Ids.length > 0) {
        const toSupersede = prevL3Ids.filter(id => id !== l3Id);
        if (toSupersede.length > 0) {
          await writeQueue.push(WRITE_PRIORITY.AGENT, () =>
            httpReq(`${QDRANT}/collections/${COLLECTION}/points/payload`, 'POST',
              { payload: { status: 'superseded', derived_to_id: l3Id }, points: toSupersede })
          );
        }
      }

      synthesized++;
      appendEvolutionLog('SYNTHESIZE',
        `L3框架: "${framework.title}" v${version} [${domain}] ← ${cluster.length}个L2洞见`
      ).catch(() => {});
      logger?.info?.(`[atlas-memory] 合成Agent: "${framework.title}" v${version} [${domain}]`);
    }
  }

  // 跨域L3合成：收集跨越多个域的L2节点群，生成跨域框架
  const crossDomainNodes = l2Nodes.filter(n =>
    !n.payload?.derived_to_id &&
    n.payload?.tags?.includes('cross-domain') &&
    (n.payload?.domain ?? '') !== (n.payload?.linked_domain ?? '')
  );
  if (crossDomainNodes.length >= CLUSTER_MIN_SIZE) {
    const crossClusters = clusterNodes(crossDomainNodes, CLUSTER_MIN_SCORE, CLUSTER_MIN_SIZE);
    for (const cluster of crossClusters) {
      const domainsInvolved = [...new Set(cluster.map(n => n.payload?.domain ?? ''))].filter(Boolean);
      const crossDomainLabel = domainsInvolved.slice(0, 3).join('×');
      const framework = await synthesizeL3(cluster, crossDomainLabel, `跨域合成：${crossDomainLabel}`);
      if (!framework) continue;
      const version   = await getNextL3Version(crossDomainLabel);
      const topic     = `${framework.title}-v${version}`;
      const sourceIds = cluster.map(n => n.id);
      const l3Path    = await writeL3Obsidian(crossDomainLabel, framework, version, sourceIds);
      const content   = `${framework.title}：${framework.summary}。原则：${(framework.principles ?? []).join('；')}`;
      const vector    = await embed(content);
      if (vector) {
        // 收集要supersede的旧版本
        let prevCrossL3Ids = [];
        if (version > 1) {
          const prevCrossR = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', {
            limit: 50, with_payload: false, with_vector: false,
            filter: { must: [{ key: 'level', match: { value: LEVEL_WISDOM } }, { key: 'domain', match: { value: crossDomainLabel } }, { key: 'status', match: { value: 'active' } }] },
          });
          prevCrossL3Ids = (prevCrossR.body?.result?.points ?? []).map(p => p.id);
        }

        let crossL3Id;
        await writeQueue.push(WRITE_PRIORITY.AGENT, async () => {
          const now = new Date().toISOString();
          const res = await upsert(vector, {
            content, category: 'work', importance: 'critical',
            tags: [...domainsInvolved, 'cross-domain-framework', `v${version}`],
            memory_type: 'distilled', created_at: now,
            source: 'synthesize-agent', session_key: 'agent',
            hit_count: 0, last_accessed_at: null, status: 'active',
            feedback_score: 1.0, level: LEVEL_WISDOM,
            domain: crossDomainLabel, topic, freshness_score: 1.0, decay_rate: 'medium',
            last_verified: now, source_ids: sourceIds,
            associated_ids: [], derived_to_id: null,
            obsidian_path: l3Path, acquisition_source: 'synthesize-agent',
            reasoning_chain:  framework.reasoning_chain  ?? '',
            counter_evidence: framework.counter_evidence ?? '',
            evidence_base:    framework.evidence_base    ?? '',
            knowledge_gaps:   framework.knowledge_gaps   ?? [],
            meta_pattern:     framework.meta_pattern     ?? '',
            challenge:        framework.challenge        ?? null,
            failure_modes:    framework.failure_modes    ?? [],
            prerequisites:    framework.prerequisites    ?? '',
          });
          crossL3Id = res?.id;
        });

        // 标记源L2节点 derived_to_id（防止下次重复合成）
        if (crossL3Id) {
          await writeQueue.push(WRITE_PRIORITY.AGENT, () =>
            httpReq(`${QDRANT}/collections/${COLLECTION}/points/payload`, 'POST',
              { payload: { derived_to_id: crossL3Id }, points: sourceIds })
          );
          // supersede旧版本
          const toSupersedeCross = prevCrossL3Ids.filter(id => id !== crossL3Id);
          if (toSupersedeCross.length > 0) {
            await writeQueue.push(WRITE_PRIORITY.AGENT, () =>
              httpReq(`${QDRANT}/collections/${COLLECTION}/points/payload`, 'POST',
                { payload: { status: 'superseded', derived_to_id: crossL3Id }, points: toSupersedeCross })
            );
          }
        }
        synthesized++;
        appendEvolutionLog('SYNTHESIZE', `跨域L3: "${topic}" [${crossDomainLabel}]`).catch(() => {});
      }
    }
  }

  // L4 元合成：收集所有 active L3，当总数 ≥ 4 且有 meta_pattern 字段时触发
  const allL3 = [];
  let l3Off = null;
  do {
    const r4 = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', {
      limit: 100, with_payload: true, with_vector: true,
      filter: { must: [{ key: 'level', match: { value: LEVEL_WISDOM } }, { key: 'status', match: { value: 'active' } }] },
      ...(l3Off ? { offset: l3Off } : {}),
    });
    if (!r4.ok) break;
    allL3.push(...(r4.body?.result?.points ?? []));
    l3Off = r4.body?.result?.next_page_offset ?? null;
  } while (l3Off);

  const l3WithPattern = allL3.filter(n => n.payload?.meta_pattern && n.payload.meta_pattern.length > 5);
  // L4冷却：从 LEVEL_META 集合中检查最近是否已合成（2小时内不重复）
  const META_L4_COOLDOWN_MS = 2 * 60 * 60 * 1000;
  const r4Cool = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', {
    limit: 10, with_payload: true, with_vector: false,
    filter: { must: [{ key: 'level', match: { value: LEVEL_META } }, { key: 'status', match: { value: 'active' } }] },
  });
  const recentL4 = (r4Cool.body?.result?.points ?? []).filter(n =>
    new Date(n.payload?.created_at ?? 0).getTime() > Date.now() - META_L4_COOLDOWN_MS
  );
  if (l3WithPattern.length >= 4 && recentL4.length === 0) {
    const meta = await synthesizeL4(l3WithPattern);
    if (meta) {
      const metaContent = `${meta.meta_title}：${meta.meta_principle}。通用行动：${meta.universal_action}`;
      const metaVector = await embed(metaContent);
      if (metaVector) {
        const now = new Date().toISOString();
        // 将跨域L3的domain字段展开去重（如"营销策略×情感学"→["营销策略","情感学"]）
        const domainsInvolved = [...new Set(
          l3WithPattern.flatMap(n => (n.payload?.domain ?? '').split('×'))
        )].filter(Boolean);
        const l4Path = await writeL4Obsidian(meta, l3WithPattern.length);
        await writeQueue.push(WRITE_PRIORITY.AGENT, () =>
          upsert(metaVector, {
            content: metaContent, level: LEVEL_META, domain: 'META',
            title: meta.meta_title ?? '',
            topic: `${meta.meta_title}-L4`, importance: 'critical',
            tags: [...domainsInvolved, 'meta-synthesis', 'L4'],
            memory_type: 'distilled', source: 'meta-synthesis',
            session_key: 'agent', status: 'active',
            feedback_score: 1.0, freshness_score: 1.0, decay_rate: 'slow',
            created_at: now, last_verified: now,
            hit_count: 0, last_accessed_at: null,
            source_ids: l3WithPattern.map(n => n.id), associated_ids: [],
            derived_to_id: null,
            obsidian_path: l4Path,
            meta_principle:           meta.meta_principle,
            domain_manifestations:    meta.domain_manifestations ?? [],
            universal_action:         meta.universal_action ?? '',
            recursive_question:       meta.recursive_question ?? '',
          })
        );
        synthesized++;
        appendEvolutionLog('META-SYNTHESIZE', `L4元规律: "${meta.meta_title}" [${domainsInvolved.join('×')}]`).catch(() => {});
        logger?.info?.(`[atlas-memory] L4元合成: "${meta.meta_title}"`);
      }
    }
  } else if (recentL4.length > 0) {
    logger?.info?.('[atlas-memory] L4元合成: 冷却中（2h内已合成），跳过');
  }

  logger?.info?.(`[atlas-memory] 合成Agent: 聚类${totalClusters}个，合成${synthesized}个L3+L4`);
  return { l2_scanned: l2Nodes.length, clusters: totalClusters, synthesized };
}

// ── Phase 7：Meta-Agent ───────────────────────────────────────────────────────

// Tavily 搜索（本地代理或云端）
const SEARCH_URL = process.env.SEARCHHARVESTER_URL ?? 'http://127.0.0.1:8000';

async function tavilySearch(query, maxResults = 5) {
  const r = await httpReq(`${SEARCH_URL}/search`, 'POST', {
    query,
    max_results: maxResults,
    search_depth: 'basic',
  }, {}, 20_000);
  if (!r.ok) return [];
  return (r.body?.results ?? []).map(item => ({
    title:   item.title ?? '',
    url:     item.url   ?? '',
    content: item.content ?? item.raw_content ?? '',
  }));
}

async function deepResearch(question, maxResults = 3) {
  const r = await httpReq(`${SEARCH_URL}/research`, 'POST', {
    query: question,
    max_results: maxResults,
    include_domains: [],
    search_depth: 'advanced',
  }, {}, 60_000);
  if (!r.ok) return null;
  return r.body?.report ?? r.body?.content ?? null;
}

function calcFreshness(createdAt, lastVerified, decayRate) {
  const halfLifeDays = DECAY_HALF_LIFE[decayRate] ?? DECAY_HALF_LIFE.medium;
  const ref  = lastVerified ?? createdAt;
  const days = (Date.now() - new Date(ref).getTime()) / 86_400_000;
  return Math.max(0, Math.pow(0.5, days / halfLifeDays));
}

async function runFreshnessUpdate(logger) {
  let offset = null;
  let updated = 0;
  let stale   = 0;
  do {
    const body = { limit: 250, with_payload: true, with_vector: false };
    if (offset != null) body.offset = offset;
    const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', body);
    if (!r.ok) break;
    const pts = r.body?.result?.points ?? [];
    for (const pt of pts) {
      const p = pt.payload ?? {};
      if (p.status === 'superseded') continue;
      const fresh = calcFreshness(p.created_at, p.last_verified, p.decay_rate ?? 'medium');
      const delta = Math.abs(fresh - (p.freshness_score ?? 1.0));
      if (delta < 0.01) continue; // skip trivial updates
      await writeQueue.push(WRITE_PRIORITY.AGENT, () =>
        qdrantPatchPayload(pt.id, { freshness_score: parseFloat(fresh.toFixed(4)) })
      );
      updated++;
      if (fresh < FRESHNESS_REFRESH) stale++;
    }
    offset = r.body?.result?.next_page_offset ?? null;
  } while (offset != null);
  logger?.info?.(`[atlas-memory] Meta-Agent 新鲜度: 更新${updated}条, 过期${stale}条`);
  return { updated, stale };
}

async function buildAcquisitionPlan(domainStats) {
  // domainStats: [{domain, l0, l1, l2, l3, staleFraction}]
  const summary = domainStats
    .filter(d => d.domain !== '未分类')
    .map(d => `${d.domain}: L0=${d.l0} L1=${d.l1} L2=${d.l2} L3=${d.l3} 过期比=${(d.staleFraction*100).toFixed(0)}%`)
    .join('\n');
  const sys = '你是知识采集规划师。严格输出JSON数组，不要解释，不要markdown代码块。';
  const user =
    `当前知识库各域状态：\n${summary}\n\n` +
    `请为知识薄弱（L1<5）或过期比高（>30%）的域生成结构化调研计划，每个计划包含多个维度的搜索查询。` +
    `输出JSON数组（最多8条）：\n` +
    `[{"domain":"域名","query":"搜索词","dimension":"该查询回答哪个维度，如：账号数据|内容规律|脚本结构|工具方法",` +
    `"completeness_check":"如何判断这个维度的信息是否完整（一句话标准）","reason":"一句话理由"}]`;
  const raw = (await deepseekGenerate(sys, user, 400)) ?? (await omlxGenerate(sys, user, 400, AGENT_OMLX_TIMEOUT_MS));
  if (!raw) return [];
  try {
    const cleaned = raw.replace(/```[a-z]*\n?/gi, '').replace(/```/g, '').trim();
    const arr = JSON.parse(cleaned);
    return Array.isArray(arr) ? arr.filter(x => x.domain && x.query).slice(0, 8) : [];
  } catch { return []; }
}

async function runDailyAcquisition(plan, logger) {
  let acquired = 0;
  const gaps = [];
  for (const item of plan) {
    const results = await tavilySearch(item.query, 3);
    const validResults = results.filter(r => r.content && r.content.length >= 100);
    if (validResults.length === 0) {
      gaps.push({ domain: item.domain, dimension: item.dimension ?? item.query, query: item.query, reason: '搜索无有效结果' });
      continue;
    }
    for (const res of validResults) {
      const content = `${res.title ? res.title + '。' : ''}${res.content}`.slice(0, 800);
      await batchStoreMemories(
        [{ content, category: 'work', importance: 'medium', memory_type: 'fact',
           tags: [item.domain, 'auto-acquired', item.dimension ?? ''].filter(Boolean), quality: 7,
           knowledge_type: 'research' }],
        `meta-agent:${res.url || item.query}`,
        undefined,
        false,
      );
      acquired++;
    }
  }
  if (gaps.length && OBSIDIAN_VAULT) {
    const gapFile = join(OBSIDIAN_VAULT, '_系统', '知识缺口.md');
    const date = new Date().toISOString().slice(0, 10);
    const lines = gaps.map(g => `- [${date}][${g.domain}] ${g.dimension}: ${g.reason}`).join('\n');
    await mkdir(join(OBSIDIAN_VAULT, '_系统'), { recursive: true });
    await appendFile(gapFile, `\n${lines}\n`, 'utf8').catch(() => {});
  }
  return acquired;
}

// 框架自我挑战 + 知识缺口驱动采集（自我演化核心）
async function runFrameworkChallengeSweep(logger) {
  // 取所有 L3 框架，按 feedback_score 倒序优先挑战评分最高的
  const l3Nodes = [];
  let off = null;
  do {
    const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', {
      limit: 50, with_payload: true, with_vector: false,
      filter: { must: [{ key: 'level', match: { value: LEVEL_WISDOM } }, { key: 'status', match: { value: 'active' } }] },
      ...(off ? { offset: off } : {}),
    });
    if (!r.ok) break;
    l3Nodes.push(...(r.body?.result?.points ?? []));
    off = r.body?.result?.next_page_offset ?? null;
  } while (off);

  // 每次只处理 knowledge_gaps 不为空的框架（最多 3 个），避免每日运行时间过长
  const toChallenge = l3Nodes
    .filter(n => (n.payload?.knowledge_gaps ?? []).length > 0)
    .sort((a, b) => (b.payload?.feedback_score ?? 1) - (a.payload?.feedback_score ?? 1))
    .slice(0, 3);

  let acquired = 0;
  for (const node of toChallenge) {
    const gaps = node.payload?.knowledge_gaps ?? [];
    const domain = node.payload?.domain ?? '通用';
    const frameTitle = node.payload?.topic ?? '未命名框架';
    const challenge = node.payload?.challenge;

    // 构建高质量搜索查询（结合缺口 + 挑战最弱点）
    const queries = [
      ...gaps.slice(0, 2),
      challenge?.attack_scenario ? `${domain} ${challenge.attack_scenario}` : null,
    ].filter(Boolean);

    logger?.info?.(`[atlas-memory] 演化: 挑战框架"${frameTitle}"，搜索${queries.length}个缺口`);

    let nodeAcquired = 0;
    for (const query of queries) {
      // 优先使用 deep research，超时则降级到普通搜索
      let content = await deepResearch(query, 3).catch(() => null);
      if (!content) {
        const results = await tavilySearch(query, 3);
        content = results
          .filter(r => r.content?.length >= 100)
          .map(r => `${r.title}：${r.content}`.slice(0, 600))
          .join('\n\n');
      }
      if (!content || content.length < 80) continue;

      // 存入 L0，携带"挑战来源"标记
      await batchStoreMemories(
        [{ content: content.slice(0, 1200), category: 'work', importance: 'high',
           memory_type: 'fact', knowledge_type: 'research',
           tags: [domain, 'challenge-search', frameTitle].filter(Boolean), quality: 7 }],
        `framework-challenge:${frameTitle}`,
        undefined,
        false,
      );
      nodeAcquired++;
    }
    acquired += nodeAcquired;

    // Clear knowledge_gaps after supplementing so this node won't be re-processed next run
    if (nodeAcquired > 0) {
      await writeQueue.push(WRITE_PRIORITY.AGENT, () =>
        qdrantPatchPayload(node.id, { knowledge_gaps: [] })
      );
    }
  }

  logger?.info?.(`[atlas-memory] 框架挑战演化: 补充${acquired}条 L0`);
  return { challenged: toChallenge.length, acquired };
}

async function writeDailyReport(date, stats) {
  if (!OBSIDIAN_VAULT) return;
  const dir = join(OBSIDIAN_VAULT, '_系统', '日报');
  await mkdir(dir, { recursive: true });
  const filename = `${date}.md`;
  const planLines = (stats.plan ?? []).map(p => `- [${p.domain}] ${p.query} — ${p.reason ?? ''}`).join('\n');
  const md = [
    '---',
    `date: ${date}`,
    `type: daily-report`,
    `generated: ${new Date().toISOString()}`,
    '---',
    '',
    `# ATLAS 日报 ${date}`,
    '',
    '## 新鲜度更新',
    `- 更新节点: ${stats.freshness?.updated ?? 0}`,
    `- 过期节点: ${stats.freshness?.stale ?? 0}`,
    '',
    '## 知识库规模',
    ...Object.entries(stats.domainStats ?? {}).map(([d, s]) =>
      `- **${d}**: L0=${s.l0} L1=${s.l1} L2=${s.l2} L3=${s.l3}`),
    '',
    '## 今日采集计划',
    planLines || '（无需补充）',
    '',
    `## 自动采集结果`,
    `- 新增 L0 记录: ${stats.acquired ?? 0} 条`,
  ].join('\n');
  await writeFile(join(dir, filename), md, 'utf8');
}

async function runMetaAgent(logger) {
  const date = new Date().toISOString().slice(0, 10);

  // 1. 新鲜度批量更新
  const freshness = await runFreshnessUpdate(logger);

  // 2. 统计各域节点分布
  const levelCountByDomain = {};
  let offset = null;
  do {
    const body = { limit: 500, with_payload: true, with_vector: false,
      filter: { must_not: [{ key: 'status', match: { value: 'superseded' } }] } };
    if (offset != null) body.offset = offset;
    const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', body);
    if (!r.ok) break;
    for (const pt of r.body?.result?.points ?? []) {
      const d = pt.payload?.domain ?? '未分类';
      if (!levelCountByDomain[d]) levelCountByDomain[d] = { l0:0, l1:0, l2:0, l3:0, staleCount:0, total:0 };
      const lvl = pt.payload?.level ?? 0;
      const key = ['l0','l1','l2','l3'][lvl] ?? 'l0';
      levelCountByDomain[d][key]++;
      levelCountByDomain[d].total++;
      if ((pt.payload?.freshness_score ?? 1.0) < FRESHNESS_REFRESH) levelCountByDomain[d].staleCount++;
    }
    offset = r.body?.result?.next_page_offset ?? null;
  } while (offset != null);

  const domainStats = Object.entries(levelCountByDomain).map(([domain, s]) => ({
    domain, l0: s.l0, l1: s.l1, l2: s.l2, l3: s.l3,
    staleFraction: s.total ? s.staleCount / s.total : 0,
  }));

  // 3. 生成采集计划
  const plan = await buildAcquisitionPlan(domainStats);
  logger?.info?.(`[atlas-memory] Meta-Agent 采集计划: ${plan.length}条`);

  // 4. 执行 web search → intakeToL0（由 searchharvester 驱动，无需外部 API Key）
  const acquired = await runDailyAcquisition(plan, logger);
  logger?.info?.(`[atlas-memory] Meta-Agent 采集: 新增${acquired}条L0`);

  // 5. 框架自我挑战演化（核心自我演化循环）
  const evolution = await runFrameworkChallengeSweep(logger);

  // 6. 写日报到 Obsidian
  await writeDailyReport(date, { freshness, domainStats: levelCountByDomain, plan, acquired, evolution });

  appendEvolutionLog('META',
    `日报: 新鲜度更新${freshness.updated}条, 过期${freshness.stale}条, 普通采集${acquired}条, 框架挑战演化${evolution.acquired}条`
  ).catch(() => {});

  return { date, freshness, domains: domainStats.length, planItems: plan.length, acquired, evolution };
}

// ── Phase 12: 结构重组Agent ───────────────────────────────────────────────────

async function fetchDomainNodeVectors(domain, limit = 100) {
  const body = {
    limit, with_payload: true, with_vector: true,
    filter: {
      must: [
        { key: 'domain', match: { value: domain } },
        { key: 'status', match: { value: 'active' } },
      ],
      must_not: [{ key: 'level', match: { value: LEVEL_RAW } }],
    },
  };
  const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', body);
  if (!r.ok) return [];
  return (r.body?.result?.points ?? []).map(p => ({
    id: p.id,
    vector: p.vector,
    obsidian_path: p.payload?.obsidian_path ?? null,
    level: p.payload?.level ?? 1,
  }));
}

function calcCohesion(vectors) {
  if (vectors.length < 2) return 1.0;
  const c = centroid(vectors);
  return vectors.reduce((sum, v) => sum + cosine(v, c), 0) / vectors.length;
}

async function patchDomainNodes(srcDomain, dstDomain) {
  let offset = null;
  const ids = [];
  do {
    const body = { limit: 256, with_payload: false, with_vector: false,
      filter: { must: [{ key: 'domain', match: { value: srcDomain } }], must_not: [{ match: { key: 'status', value: 'superseded' } }] } };
    if (offset) body.offset = offset;
    const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', body);
    if (!r.ok) break;
    ids.push(...(r.body?.result?.points ?? []).map(p => p.id));
    offset = r.body?.result?.next_page_offset ?? null;
  } while (offset);

  for (let i = 0; i < ids.length; i += 100) {
    await httpReq(`${QDRANT}/collections/${COLLECTION}/points/payload`, 'POST', {
      payload: { domain: dstDomain }, points: ids.slice(i, i + 100),
    });
  }
  return ids.length;
}

async function moveObsidianDomainFiles(srcDir, dstDir, logger) {
  if (!OBSIDIAN_VAULT || !srcDir || !dstDir) return;
  const src = join(OBSIDIAN_VAULT, srcDir);
  const dst = join(OBSIDIAN_VAULT, dstDir);
  try {
    await mkdir(dst, { recursive: true });
    const entries = await readdir(src, { withFileTypes: true }).catch(() => []);
    for (const e of entries) {
      if (e.name.startsWith('.')) continue;
      const s = join(src, e.name), d = join(dst, e.name);
      await rename(s, d).catch(async (err) => {
        if (err.code === 'EXDEV') {
          const buf = await readFile(s).catch(() => null);
          if (buf) { await writeFile(d, buf).catch(() => {}); await unlink(s).catch(() => {}); }
        }
      });
    }
    await rmdir(src).catch(() => {});
  } catch (e) {
    logger?.info?.(`[atlas-memory] 重组：文件移动失败 ${e.message}`);
  }
}

async function writeRestructureLog(date, actions) {
  if (!OBSIDIAN_VAULT || !actions.length) return;
  const dir = join(OBSIDIAN_VAULT, '_系统', '重组日志');
  await mkdir(dir, { recursive: true });
  const content = `# 结构重组日志 ${date}\n\n${actions.map(a => `- ${a}`).join('\n')}\n`;
  await writeFile(join(dir, `${date}.md`), content, 'utf8').catch(() => {});
}

// ── L1 知识补全 Agent ──────────────────────────────────────────────────────────
// 扫描低完整度 L1 节点，用 searcharvester + DeepSeek 补全缺失字段，写回 Qdrant + vault
async function runL1CompletionAgent(logger) {
  // 1. 拉取 completeness_score < 0.85 且 completeness_gaps 非空的 L1 节点
  const candidates = [];
  let offset = null;
  do {
    const body = {
      limit: 200, with_payload: true, with_vector: false,
      filter: {
        must: [
          { key: 'level',  match: { value: LEVEL_KNOWLEDGE } },
          { key: 'status', match: { value: 'active' } },
        ],
        must_not: [
          { key: 'record_type', match: { value: 'entity' } },
          { key: 'record_type', match: { value: 'relation' } },
        ],
        // null值不匹配range过滤，需用should捕获null记录
        should: [
          { key: 'completeness_score', range: { lt: 0.85 } },
          { key: 'completeness_score', is_null: true },
        ],
      },
    };
    if (offset != null) body.offset = offset;
    const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', body);
    if (!r.ok) break;
    for (const pt of r.body?.result?.points ?? []) {
      const gaps = pt.payload?.completeness_gaps ?? [];
      if (gaps.length > 0) candidates.push(pt);
    }
    offset = r.body?.result?.next_page_offset ?? null;
  } while (offset != null);

  if (!candidates.length) {
    logger?.info?.('[L1Completion] 无需补全的节点');
    return { processed: 0, filled: 0 };
  }

  // 按完整度升序排，优先处理最不完整的，每次最多 5 个节点
  candidates.sort((a, b) => (a.payload?.completeness_score ?? 0) - (b.payload?.completeness_score ?? 0));
  const batch = candidates.slice(0, 20);
  logger?.info?.(`[L1Completion] 待补全节点 ${candidates.length} 个，本次处理 ${batch.length} 个`);

  // gap 字段名 → 中文说明（用于构造搜索词和 prompt）
  const GAP_META = {
    rule_statement:        { label: '原则表述',   hint: '这个规律/原则的准确表述是什么' },
    rationale:             { label: '底层逻辑',   hint: '为什么这个规律有效，底层心理或社会学原理' },
    applicable_scenarios:  { label: '适用场景',   hint: '哪些场景和条件下可以使用' },
    exceptions:            { label: '例外情况',   hint: '什么情况下不适用，需要注意什么' },
    examples:              { label: '具体例子',   hint: '有哪些真实具体的例子或案例' },
    definition:            { label: '定义',       hint: '准确的定义是什么' },
    scope:                 { label: '适用范围',   hint: '适用的范围和边界' },
    related_concepts:      { label: '关联概念',   hint: '相关的概念有哪些' },
    claim:                 { label: '核心论点',   hint: '核心主张是什么' },
    reasoning:             { label: '推理链条',   hint: '推理逻辑和论证过程' },
    evidence:              { label: '支撑证据',   hint: '有哪些证据支撑' },
    limitations:           { label: '局限性',     hint: '有哪些局限和不足' },
    steps:                 { label: '操作步骤',   hint: '具体的操作步骤是什么' },
    preconditions:         { label: '前提条件',   hint: '需要满足哪些前提' },
    expected_outcome:      { label: '预期结果',   hint: '执行后预期的结果是什么' },
    edge_cases:            { label: '边界情况',   hint: '边界情况和特殊情形如何处理' },
    statement:             { label: '事实陈述',   hint: '事实的准确陈述' },
    source_context:        { label: '来源背景',   hint: '来源和背景信息' },
    temporal_scope:        { label: '时效性',     hint: '何时有效，是否会过时' },
    confidence:            { label: '可信度',     hint: '可信度如何，理由是什么' },
    summary:               { label: '摘要',       hint: '核心内容摘要' },
    // video_script 专用
    hook:                  { label: '钩子/开场',   hint: '视频开头如何吸引注意，钩子句是什么' },
    structure:             { label: '内容结构',    hint: '视频整体结构和各部分如何组织' },
    pain_points:           { label: '痛点/价值点', hint: '目标受众的核心痛点或价值主张' },
    cta:                   { label: '行动呼吁',    hint: '结尾引导观众做什么（关注/评论/购买）' },
    // sop 专用
    tools_required:        { label: '所需工具',    hint: '完成这个流程需要哪些工具或资源' },
  };

  let totalFilled = 0;

  for (const pt of batch) {
    const payload   = pt.payload ?? {};
    const topic     = payload.topic     ?? '未命名';
    const domain    = payload.domain    ?? '通用';
    const ct        = payload.content_type ?? 'principle';
    const gaps      = payload.completeness_gaps ?? [];
    const existingContent = [
      payload.summary, payload.rule_statement, payload.rationale,
      payload.applicable_scenarios, payload.exceptions,
      ...(payload.examples ?? []), ...(payload.key_points ?? []),
    ].filter(Boolean).join('\n');

    const updates = {};
    let filledCount = 0;

    for (const gap of gaps.slice(0, 3)) {  // 每节点最多补 3 个 gap
      const meta = GAP_META[gap];
      if (!meta) continue;

      // 优先使用节点自身的完整内容作为上下文（适用于课程/书籍等私有知识）
      const internalCtx = [
        payload.content,
        payload.summary,
        payload.rule_statement, payload.rationale,
        payload.applicable_scenarios, payload.exceptions,
        ...(payload.key_points ?? []),
        ...(payload.examples ?? []).map(e => typeof e === 'string' ? e : JSON.stringify(e)),
      ].filter(Boolean).join('\n\n').slice(0, 2500);

      let context = internalCtx;
      // 仅当内部内容过短时才调用外部搜索补充
      if (internalCtx.length < 200) {
        const query = `${topic} ${meta.hint} ${domain}`;
        const results = await tavilySearch(query, 3).catch(() => []);
        const external = results
          .filter(r => (r.content?.length ?? 0) >= 60)
          .map(r => `${r.title}：${r.content}`.slice(0, 400))
          .join('\n\n');
        if (external) context = (internalCtx + '\n\n' + external).slice(0, 2500);
      }

      if (!context || context.length < 50) continue;

      const sys = '你是知识整理专家。根据已有知识内容，为知识节点补全缺失字段。只输出补全内容，不加解释，内容简洁准确，必须基于已有内容推导，不得编造。';
      const user = `知识节点：${topic}（域：${domain}，类型：${ct}）

已有知识内容：
${context}

需要补全的字段：【${meta.label}】（${meta.hint}）

请直接输出【${meta.label}】字段的内容（${gap === 'examples' ? 'JSON数组格式，如["例子1","例子2"]' : '20-60字文本'}）：`;

      const filled = (await deepseekGenerate(sys, user, 300))?.trim();
      if (!filled || filled.length < 5) continue;

      // 处理数组类型字段
      if (['examples', 'related_concepts', 'evidence', 'steps', 'key_points'].includes(gap)) {
        try {
          const arr = JSON.parse(filled.match(/\[[\s\S]*\]/)?.[0] ?? '[]');
          if (Array.isArray(arr) && arr.length > 0) {
            updates[gap] = arr.map(s => String(s).trim()).filter(Boolean);
            filledCount++;
          }
        } catch { /* skip malformed */ }
      } else {
        updates[gap] = filled.replace(/^["']|["']$/g, '').trim();
        filledCount++;
      }
    }

    if (!filledCount) continue;

    // 合并更新，重算完整度
    const merged = { ...payload, ...updates };
    const { score: newScore, gaps: newGaps } = calcCompleteness(merged, ct);
    updates.completeness_score = parseFloat(newScore.toFixed(2));
    updates.completeness_gaps  = newGaps;

    // 写回 Qdrant
    await writeQueue.push(WRITE_PRIORITY.AGENT, () => qdrantPatchPayload(pt.id, updates));

    // 更新 vault markdown（如果有路径）
    const vaultPath = payload.obsidian_path;
    if (OBSIDIAN_VAULT && vaultPath) {
      await writeL1Obsidian(domain, topic, merged, payload.source_l0 ?? null)
        .catch(() => null);
    }

    totalFilled += filledCount;
    logger?.info?.(`[L1Completion] ${topic}：补全 ${filledCount} 个字段，完整度 ${((payload.completeness_score ?? 0) * 100).toFixed(0)}% → ${(newScore * 100).toFixed(0)}%`);
  }

  logger?.info?.(`[L1Completion] 完成，共补全 ${totalFilled} 个字段`);
  return { processed: batch.length, filled: totalFilled };
}

async function runRestructureAgent(logger) {
  // ── 步骤0：弱分类节点重新归类（未分类 + domain_score 低于阈值）──────────────
  let reassigned = 0;
  try {
    let rcOffset = null;
    const weakNodes = [];
    do {
      const rr = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', {
        limit: 200, with_payload: true, with_vector: true,
        filter: {
          should: [
            { key: 'domain', match: { value: '未分类' } },
            { key: 'domain', is_null: true },
          ],
          must: [
            { key: 'level', match: { value: LEVEL_KNOWLEDGE } },
            { key: 'status', match: { value: 'active' } },
          ],
        },
        ...(rcOffset != null ? { offset: rcOffset } : {}),
      });
      if (!rr.ok) break;
      weakNodes.push(...(rr.body?.result?.points ?? []).filter(p => p.vector));
      rcOffset = rr.body?.result?.next_page_offset ?? null;
    } while (rcOffset != null);

    logger?.info?.(`[atlas-memory] 重组Agent: 弱分类节点${weakNodes.length}条，尝试重新归类`);
    for (const node of weakNodes) {
      const match = await matchDomainForVector(node.vector);
      if (match.domain && match.score >= DOMAIN_MATCH_SCORE) {
        await writeQueue.push(WRITE_PRIORITY.AGENT, () =>
          qdrantPatchPayload(node.id, {
            domain:       match.domain,
            domain_score: parseFloat(match.score.toFixed(3)),
            last_verified: new Date().toISOString(),
          })
        );
        reassigned++;
      }
    }
    if (reassigned) logger?.info?.(`[atlas-memory] 重组Agent: 重新归类${reassigned}条`);
  } catch (e) {
    logger?.warn?.(`[atlas-memory] 重组Agent: 弱分类重归失败 ${e.message}`);
  }

  const domains = Object.keys(DOMAIN_DIRS);
  if (domains.length < 2) {
    logger?.info?.('[atlas-memory] 重组Agent: 域数量不足，跳过');
    return { merges: 0, splits: 0, reassigned };
  }

  logger?.info?.('[atlas-memory] 重组Agent: 开始结构分析...');

  // 获取各域节点向量
  const domainData = {};
  await Promise.all(domains.map(async d => {
    const nodes = await fetchDomainNodeVectors(d, 100);
    const vecs = nodes.map(n => n.vector).filter(Boolean);
    domainData[d] = { nodes, vecs, count: nodes.length };
  }));

  const active = domains.filter(d => domainData[d].count >= RESTRUCTURE_MIN_NODES);
  logger?.info?.(`[atlas-memory] 重组Agent: 活跃域 ${active.length}/${domains.length}`);

  const actions = [];
  let merges = 0, splits = 0;
  const date = new Date().toISOString().slice(0, 10);

  // ── 1. 合并检测 ────────────────────────────────────────────────────
  const cents = {};
  for (const d of active) {
    if (domainData[d].vecs.length >= 2) cents[d] = centroid(domainData[d].vecs);
  }

  const mergeCandidates = [];
  const dList = active.filter(d => cents[d]);
  for (let i = 0; i < dList.length; i++) {
    for (let j = i + 1; j < dList.length; j++) {
      const sim = cosine(cents[dList[i]], cents[dList[j]]);
      if (sim >= MERGE_SIM_THRESHOLD)
        mergeCandidates.push({ a: dList[i], b: dList[j], sim });
    }
  }
  mergeCandidates.sort((x, y) => y.sim - x.sim);

  const merged = new Set();
  for (const mc of mergeCandidates) {
    if (merges >= RESTRUCTURE_MAX_MERGES) break;
    if (merged.has(mc.a) || merged.has(mc.b)) continue;

    const [src, dst] = domainData[mc.a].count <= domainData[mc.b].count
      ? [mc.a, mc.b] : [mc.b, mc.a];

    logger?.info?.(`[atlas-memory] 重组：合并 "${src}" → "${dst}" sim=${mc.sim.toFixed(3)}`);
    const patched = await patchDomainNodes(src, dst);
    await moveObsidianDomainFiles(DOMAIN_DIRS[src], DOMAIN_DIRS[dst], logger);
    await writeDomainIndex(dst, { l0: 0, l1: domainData[dst].count + patched, l2: 0, l3: 0 }).catch(() => {});

    delete DOMAIN_DIRS[src];
    delete DOMAIN_DESCRIPTIONS[src];
    domainEmbeddingCache.delete(src);

    appendEvolutionLog('RESTRUCTURE', `合并: "${src}" → "${dst}" ${patched}条 sim=${mc.sim.toFixed(3)}`).catch(() => {});
    actions.push(`合并: \`${src}\`(${domainData[src].count}条) → \`${dst}\` (相似度=${mc.sim.toFixed(3)})`);
    merged.add(src);
    merges++;
  }

  // ── 2. 分裂检测 ────────────────────────────────────────────────────
  const splitCandidates = [];
  for (const d of active) {
    if (merged.has(d)) continue;
    const vecs = domainData[d].vecs;
    if (vecs.length < SPLIT_MIN_NODES) continue;
    const cohesion = calcCohesion(vecs);
    if (cohesion < SPLIT_COHESION_THRESHOLD)
      splitCandidates.push({ domain: d, cohesion, nodes: domainData[d].nodes });
  }
  splitCandidates.sort((x, y) => x.cohesion - y.cohesion);

  for (const sc of splitCandidates) {
    if (splits >= RESTRUCTURE_MAX_SPLITS) break;

    const items = sc.nodes.filter(n => n.vector).map(n => ({ ...n, embedding: n.vector }));
    const clusters = clusterNodes(items, 0.72, 5);
    if (clusters.length < 2) continue;

    // DeepSeek 命名新子域
    const sampleA = clusters[0].slice(0, 3).map(n => n.obsidian_path?.split('/').pop() ?? n.id).join(', ');
    const sampleB = clusters[1].slice(0, 3).map(n => n.obsidian_path?.split('/').pop() ?? n.id).join(', ');
    const prompt = `域名:"${sc.domain}" 内聚度过低需分裂。\n子域A样本:${sampleA}\n子域B样本:${sampleB}\n为两子域各取简洁中文名(4-8字)。只输出JSON:{"nameA":"...","nameB":"..."}`;

    let nameA = `${sc.domain}-A`, nameB = `${sc.domain}-B`;
    try {
      const splitSys = '你是知识库结构优化专家。严格只输出JSON对象，不要解释。';
      const rawName = await deepseekGenerate(splitSys, prompt, 80) ?? await omlxGenerate(splitSys, prompt, 80, 15_000);
      if (rawName) {
        const m = rawName.match(/\{[^}]+\}/);
        if (m) { const p = JSON.parse(m[0]); nameA = p.nameA || nameA; nameB = p.nameB || nameB; }
      }
    } catch (_) {}

    // 生成子域描述（用于向量匹配精度）
    let descA = `${sc.domain}子域A：${sampleA}`, descB = `${sc.domain}子域B：${sampleB}`;
    try {
      const descSys = '你是知识库结构专家。为知识子域生成精确的中文描述（15-30字），用于语义匹配。严格只输出JSON：{"descA":"...","descB":"..."}';
      const descPrompt = `原域:"${sc.domain}" 分裂为两子域\n子域"${nameA}"样本:${sampleA}\n子域"${nameB}"样本:${sampleB}\n为两子域各生成15-30字描述`;
      const rawDesc = await deepseekGenerate(descSys, descPrompt, 100) ?? await omlxGenerate(descSys, descPrompt, 100, 15_000);
      if (rawDesc) {
        const m = rawDesc.match(/\{[^}]+\}/);
        if (m) { const p = JSON.parse(m[0]); descA = p.descA || descA; descB = p.descB || descB; }
      }
    } catch (_) {}

    logger?.info?.(`[atlas-memory] 重组：分裂 "${sc.domain}" → "${nameA}" + "${nameB}" cohesion=${sc.cohesion.toFixed(3)}`);

    await createDomainStructure(nameA, { description: descA, dimensions: [], keywords: [] });
    await createDomainStructure(nameB, { description: descB, dimensions: [], keywords: [] });

    // Register new domains in runtime caches
    DOMAIN_DIRS[nameA] = nameA;
    DOMAIN_DIRS[nameB] = nameB;
    DOMAIN_DESCRIPTIONS[nameA] = descA;
    DOMAIN_DESCRIPTIONS[nameB] = descB;
    const vecA = await embed(DOMAIN_DESCRIPTIONS[nameA]);
    if (vecA) domainEmbeddingCache.set(nameA, vecA);
    const vecB = await embed(DOMAIN_DESCRIPTIONS[nameB]);
    if (vecB) domainEmbeddingCache.set(nameB, vecB);

    const patchBatch = async (ids, domain) => {
      for (let i = 0; i < ids.length; i += 100)
        await httpReq(`${QDRANT}/collections/${COLLECTION}/points/payload`, 'POST',
          { payload: { domain }, points: ids.slice(i, i + 100) });
    };
    const clusterAIds = new Set(clusters[0].map(n => String(n.id)));
    const clusterBIds = new Set(clusters[1].map(n => String(n.id)));
    await patchBatch(clusters[0].map(n => n.id), nameA);
    await patchBatch(clusters[1].map(n => n.id), nameB);

    // Patch residual nodes (not in either cluster) to nameA
    const residualIds = domainData[sc.domain].nodes
      .filter(n => !clusterAIds.has(String(n.id)) && !clusterBIds.has(String(n.id)))
      .map(n => n.id);
    if (residualIds.length > 0) await patchBatch(residualIds, nameA);

    // Move Obsidian files: old domain → nameA (nameB gets new files from Organize Agent)
    if (OBSIDIAN_VAULT) {
      for (const level of ['L1', 'L2', 'L3']) {
        const srcDir = join(OBSIDIAN_VAULT, level, sc.domain);
        const dstDir = join(OBSIDIAN_VAULT, level, nameA);
        await mkdir(dstDir, { recursive: true });
        const entries = await readdir(srcDir, { withFileTypes: true }).catch(() => []);
        for (const e of entries) {
          if (e.name.startsWith('.')) continue;
          const s = join(srcDir, e.name), d = join(dstDir, e.name);
          await rename(s, d).catch(async (err) => {
            if (err.code === 'EXDEV') {
              const buf = await readFile(s).catch(() => null);
              if (buf) { await writeFile(d, buf).catch(() => {}); await unlink(s).catch(() => {}); }
            }
          });
        }
        await rmdir(srcDir).catch(() => {});
      }
    }

    delete DOMAIN_DIRS[sc.domain];
    delete DOMAIN_DESCRIPTIONS[sc.domain];
    domainEmbeddingCache.delete(sc.domain);

    appendEvolutionLog('RESTRUCTURE', `分裂: "${sc.domain}" → "${nameA}"+"${nameB}" cohesion=${sc.cohesion.toFixed(3)}`).catch(() => {});
    actions.push(`分裂: \`${sc.domain}\`(${sc.nodes.length}条) → \`${nameA}\` + \`${nameB}\` (内聚度=${sc.cohesion.toFixed(3)})`);
    splits++;
  }

  if (actions.length) await writeRestructureLog(date, actions).catch(() => {});
  logger?.info?.(`[atlas-memory] 重组Agent完成: 合并${merges}次, 分裂${splits}次`);
  return { merges, splits, reassigned, actions };
}

// ── 工具结果格式 ──────────────────────────────────────────────────────────────
function jsonResult(payload) {
  return { content: [{ type: 'text', text: JSON.stringify(payload, null, 2) }] };
}

// ── Phase 10: MCP Server ──────────────────────────────────────────────────────

async function getLevelStats() {
  const counts = { 0: 0, 1: 0, 2: 0, 3: 0 };
  let offset = null;
  do {
    const body = { limit: 256, with_payload: true, with_vector: false,
      filter: { must_not: [{ key: 'status', match: { value: 'superseded' } }] } };
    if (offset) body.offset = offset;
    const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', body);
    if (!r.ok) break;
    for (const p of r.body?.result?.points ?? []) {
      const lv = p.payload?.level ?? 0;
      if (lv in counts) counts[lv]++;
    }
    offset = r.body?.result?.next_page_offset ?? null;
  } while (offset);
  return counts;
}

const MCP_TOOL_DEFS = [
  {
    name: 'atlas_recall',
    description: '语义检索知识库，返回最相关的 top-K 条目（含层级、域、重要度）',
    inputSchema: {
      type: 'object', required: ['query'],
      properties: {
        query:     { type: 'string',  description: '自然语言查询' },
        limit:     { type: 'integer', default: 5, minimum: 1, maximum: 20 },
        min_score: { type: 'number',  default: 0.45 },
        domain:    { type: 'string',  description: '限定域过滤，如 "情感学"，不传则全库搜索' },
      },
    },
  },
  {
    name: 'atlas_store',
    description: '向知识库写入一条 L0 原料，自动触发整理 Agent 升级为 L1',
    inputSchema: {
      type: 'object', required: ['content'],
      properties: {
        content:     { type: 'string' },
        topic:       { type: 'string', description: '标题/主题，用作文件名，不填则自动从内容提取' },
        domain:      { type: 'string', description: '所属域，如"情感学"，不填则由整理Agent分类' },
        importance:  { type: 'string', enum: ['low', 'medium', 'high', 'critical'], default: 'medium' },
        memory_type: { type: 'string', enum: ['fact', 'principle', 'case', 'method', 'quote'], default: 'fact' },
        tags:        { type: 'array',  items: { type: 'string' }, default: [] },
      },
    },
  },
  {
    name: 'atlas_stats',
    description: '查看知识库状态：L0-L3 各层计数、域分布、服务健康',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'atlas_evolve',
    description: '手动触发全套 Agent pipeline（整理→域检测→关联→合成→Meta）',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'atlas_feedback',
    description: '对某条记忆打分，提升或降低其权重',
    inputSchema: {
      type: 'object', required: ['id', 'verdict'],
      properties: {
        id:      { type: 'string', description: '记忆 ID（从 atlas_recall 获取）' },
        verdict: { type: 'string', enum: ['correct', 'wrong', 'outdated'] },
      },
    },
  },
  {
    name: 'atlas_timeline',
    description: '按标签/域查看知识演进时间线（L0→L1→L2→L3 成长轨迹）。不填 tag 则返回最近全局条目。',
    inputSchema: {
      type: 'object',
      properties: {
        tag:   { type: 'string', description: '标签名或域名，不填则查全局最新' },
        limit: { type: 'integer', default: 20, maximum: 100 },
        order: { type: 'string',  enum: ['asc', 'desc'], default: 'desc' },
      },
    },
  },
];

async function mcpExecute(toolName, args) {
  switch (toolName) {

    case 'atlas_intake': {
      const { content, content_type = 'other', group_id, group_total, chunk_index,
        source_meta, domain, topic, tags = [], importance = 'medium' } = args;
      if (!content?.trim()) throw new Error('content 不能为空');
      const r = await intakeToL0({ content, content_type, group_id, group_total, chunk_index,
        source_meta, domain, topic, tags: Array.isArray(tags) ? tags : [],
        importance, source: 'mcp_intake' });
      if (!r.ok) throw new Error(r.error ?? 'intake 失败');
      return { ok: true, id: r.id, group_id: r.group_id ?? null,
        is_fragment: !!(r.group_id), content_type };
    }

    case 'atlas_recall': {
      const { query, limit = 5, min_score = SCORE_MIN, domain, source_type,
        intent = 'relevant', min_confidence = 0, expand_entities = false } = args;
      if (!query?.trim()) throw new Error('query 不能为空');
      const vector = await embed(query);
      if (!vector) throw new Error('embed 不可用');
      const hits = await qdrantSearch(vector, {
        limit: expand_entities ? limit * 2 : limit,
        minScore: min_score, domain, source_type, intent, min_confidence, expand_entities,
      });
      const sorted = applyTimeDecay(hits, intent).slice(0, limit);
      trackAccess(sorted).catch(() => {});
      return sorted.map(h => ({
        id: h.id, score: h.effectiveScore ?? h.score,
        level: h.payload?.level ?? 0,
        domain: h.payload?.domain ?? null,
        topic: h.payload?.topic ?? null,
        content: h.payload?.content,
        memory_type: h.payload?.memory_type,
        importance: h.payload?.importance,
        confidence: h.payload?.confidence ?? null,
        faithfulness_score: h.payload?.faithfulness_score ?? null,
        entity_ids: h.payload?.entity_ids ?? [],
        source_type: h.payload?.source_type ?? null,
        created_at: h.payload?.created_at,
      }));
    }

    case 'atlas_store': {
      const { content, importance = 'medium', memory_type = 'fact', tags = [], topic, domain,
        knowledge_type = 'capture', content_type, group_id, source_meta, platform } = args;
      if (!content?.trim()) throw new Error('content 不能为空');
      const r = await intakeToL0({ content, importance, memory_type, tags, topic, domain,
        source: 'mcp', knowledge_type, content_type, group_id,
        source_meta: source_meta ? { ...source_meta, platform: source_meta.platform ?? platform } : (platform ? { platform } : null) });
      if (!r.ok) throw new Error(r.error ?? 'store 失败');
      return { ok: true, id: r.id, deduplicated: r.deduplicated ?? false };
    }

    case 'atlas_stats': {
      const [qdrantRes, levels] = await Promise.all([
        httpReq(`${QDRANT}/collections/${COLLECTION}`),
        getLevelStats(),
      ]);
      const countByType = async (rt) => {
        const res = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/count`, 'POST',
          { filter: { must: [{ key: 'record_type', match: { value: rt } }] }, exact: false });
        return res.ok ? (res.body?.result?.count ?? 0) : 0;
      };
      const [entity_count, relation_count] = await Promise.all([
        countByType(RECORD_TYPES.ENTITY), countByType(RECORD_TYPES.RELATION),
      ]);
      const total = qdrantRes.body?.result?.points_count ?? 0;
      return {
        version: '12.0.0',
        total, entity_count, relation_count,
        knowledge_count: Math.max(0, total - entity_count - relation_count),
        levels: { L0: levels[0], L1: levels[1], L2: levels[2], L3: levels[3] },
        domains: Object.keys(DOMAIN_DESCRIPTIONS),
        qdrant_ok: qdrantRes.ok, deepseek_ok: !!DEEPSEEK_API_KEY, mcp_port: MCP_PORT,
      };
    }

    case 'atlas_evolve': {
      const r = await runEvolution(null);
      // 通过 runAgent 锁触发，防止与定时器产生并发实例
      runAgent('organize',    () => runOrganizeAgent(null)).catch(() => {});
      runAgent('associate',   () => runAssociateAgent(null)).catch(() => {});
      runAgent('synthesize',  () => runSynthesizeAgent(null)).catch(() => {});
      runAgent('completion',  () => runL1CompletionAgent(null)).catch(() => {});
      return { ok: true, ...r, agents_triggered: ['organize', 'associate', 'synthesize', 'completion'] };
    }

    case 'atlas_feedback': {
      const { id, verdict } = args;
      if (!id || !verdict) throw new Error('id 和 verdict 必填');
      // Qdrant numeric IDs must be numbers, not strings
      const parsedId = /^\d+$/.test(String(id)) ? Number(id) : id;
      const getR = await httpReq(`${QDRANT}/collections/${COLLECTION}/points`, 'POST',
        { ids: [parsedId], with_payload: true, with_vector: false });
      const point = getR.body?.result?.[0];
      if (!point) throw new Error('记忆不存在');
      const cur = point.payload?.feedback_score ?? 1.0;
      const delta = verdict === 'correct' ? FEEDBACK_BOOST : -FEEDBACK_DECAY;
      const next = Math.max(0, Math.min(1, cur + delta));
      if (next <= FEEDBACK_DELETE_FLOOR) {
        await qdrantDelete([parsedId]);
        return { action: 'deleted', feedback_score: next };
      }
      await qdrantPatchPayload(parsedId, { feedback_score: next });
      return { action: 'updated', verdict, feedback_score: { before: cur, after: next } };
    }

    case 'atlas_timeline': {
      const { tag, limit = 20, order = 'desc' } = args;
      const tagStr = tag?.trim() ?? '';
      let offset = null; const pts = [];
      do {
        const mustNot = [{ key: 'status', match: { value: 'superseded' } }];
        const must = tagStr
          ? [{ key: 'tags', match: { any: [tagStr] } }]
          : [];
        const body = { limit: 100, with_payload: true, with_vector: false,
          filter: { must, must_not: mustNot } };
        if (offset) body.offset = offset;
        const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', body);
        if (!r.ok) break;
        pts.push(...(r.body?.result?.points ?? []));
        offset = r.body?.result?.next_page_offset ?? null;
      } while (offset && pts.length < 500);
      const sorted = pts.sort((a, b) => {
        const ta = new Date(a.payload?.created_at ?? 0).getTime();
        const tb = new Date(b.payload?.created_at ?? 0).getTime();
        return order === 'asc' ? ta - tb : tb - ta;
      }).slice(0, limit);
      return sorted.map(p => ({
        id: p.id, level: p.payload?.level ?? 0,
        domain: p.payload?.domain ?? null,
        topic: p.payload?.topic ?? null,
        date: (p.payload?.created_at ?? '').slice(0, 10),
        content: (p.payload?.content ?? '').slice(0, 100),
        memory_type: p.payload?.memory_type,
      }));
    }

    default:
      throw Object.assign(new Error(`未知工具: ${toolName}`), { code: -32601 });
  }
}

function startMcpServer(logger) {
  const server = http.createServer((req, res) => {
    if (req.method === 'GET' && req.url === '/health') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true, name: 'atlas-memory', version: '11.0.0' }));
      return;
    }
    if (req.method !== 'POST') { res.writeHead(405); res.end(); return; }

    let body = '';
    req.on('data', c => { body += c; if (body.length > 1_048_576) req.destroy(); });
    req.on('end', async () => {
      let rpcId = null;
      const send = (payload) => {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(payload));
      };
      try {
        const rpc = JSON.parse(body);
        rpcId = rpc.id ?? null;
        const { method, params } = rpc;

        if (method === 'initialize') {
          return send({ jsonrpc: '2.0', id: rpcId, result: {
            protocolVersion: '2024-11-05',
            capabilities: { tools: {} },
            serverInfo: { name: 'atlas-memory', version: '11.0.0' },
          }});
        }
        if (method === 'notifications/initialized') {
          res.writeHead(204); res.end(); return;
        }
        if (method === 'tools/list') {
          return send({ jsonrpc: '2.0', id: rpcId, result: { tools: MCP_TOOL_DEFS } });
        }
        if (method === 'tools/call') {
          const { name: toolName, arguments: toolArgs } = params ?? {};
          const result = await mcpExecute(toolName, toolArgs ?? {});
          return send({ jsonrpc: '2.0', id: rpcId, result: {
            content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
          }});
        }
        send({ jsonrpc: '2.0', id: rpcId,
          error: { code: -32601, message: `Method not found: ${method}` } });
      } catch (e) {
        send({ jsonrpc: '2.0', id: rpcId,
          error: { code: e.code ?? -32603, message: e.message } });
      }
    });
  });

  server.on('error', e => logger?.info?.(`[atlas-memory] MCP Server 错误: ${e.message}`));
  server.listen(MCP_PORT, '127.0.0.1', () =>
    logger?.info?.(`[atlas-memory] MCP Server 就绪，端口 ${MCP_PORT}（claude_desktop_config: url=http://127.0.0.1:${MCP_PORT}）`)
  );
}

// ── 插件注册 ──────────────────────────────────────────────────────────────────
export const name        = 'atlas-memory';
// ── 旧域名迁移（一次性，幂等）──────────────────────────────────────────────────
const LEGACY_DOMAIN_MAP = {
  '01-情感学':        '情感学',
  '02-营销学-刘克亚': '营销',
  '03-营销学-智多星': '营销',
  '04-战略-刘海峰':   '战略',
  '营销(科特勒)':     '营销',
};

async function runLegacyDomainMigration(logger) {
  // 快速计数：若无旧域节点则跳过
  const should = Object.keys(LEGACY_DOMAIN_MAP).map(d => ({ key: 'domain', match: { value: d } }));
  const cr = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/count`, 'POST',
    { filter: { should } });
  if (!cr.ok || (cr.body?.result?.count ?? 0) === 0) return 0;

  let total = 0;
  for (const [oldDomain, newDomain] of Object.entries(LEGACY_DOMAIN_MAP)) {
    let offset = null;
    const ids = [];
    do {
      const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', {
        limit: 500, with_payload: false, with_vector: false,
        filter: { must: [{ key: 'domain', match: { value: oldDomain } }] },
        ...(offset != null ? { offset } : {}),
      });
      if (!r.ok) break;
      ids.push(...(r.body?.result?.points ?? []).map(p => p.id));
      offset = r.body?.result?.next_page_offset ?? null;
    } while (offset != null);

    if (!ids.length) continue;

    // 按 100 条一批 patch，保证幂等
    for (let i = 0; i < ids.length; i += 100) {
      await httpReq(`${QDRANT}/collections/${COLLECTION}/points/payload`, 'POST', {
        payload: { domain: newDomain },
        points: ids.slice(i, i + 100),
      });
    }
    total += ids.length;
    logger?.info?.(`[atlas-memory] 域迁移: "${oldDomain}" → "${newDomain}" (${ids.length}条)`);
  }
  if (total) logger?.info?.(`[atlas-memory] 域迁移完成，共${total}条`);
  return total;
}

export const description = 'ATLAS Memory v12.0.0 — 多模态自主演化知识系统（L0-L3四层 · 实体注册 · 关系图谱 · 多域并行 · DeepSeek提炼 · 即时录入 · 5Agent全自治 · MCP Server）';

export function register(api) {
  const logger = api.logger;

  startMcpServer(logger);

  // 恢复持久化运行时状态
  readFile(STATE_FILE, 'utf8').then(raw => {
    try { const s = JSON.parse(raw); lastAssociateRun = s.lastAssociateRun ?? 0; } catch {}
  }).catch(() => {});

  ensureCollection()
    .then(() => migrateSchema(logger))
    .then(() => restoreDynamicDomains(logger))
    .then(() => runLegacyDomainMigration(logger))
    .catch(() => {});
  setInterval(() => runEvolution(logger).catch(() => {}), 24 * 60 * 60 * 1000);
  setInterval(() => backupCollection(logger).catch(() => {}), 7 * 24 * 60 * 60 * 1000);
  // Phase 3：整理Agent（1h 周期 + 启动后 10s 首次触发）
  setInterval(() => runAgent('organize', () => runOrganizeAgent(logger)).catch(() => {}), ORGANIZE_INTERVAL_MS);
  setTimeout(() => runAgent('organize', () => runOrganizeAgent(logger)).catch(() => {}), 10_000);
  // Phase 4：域检测Agent（6h 周期 + 启动后 30s 首次触发）
  setInterval(() => runAgent('domain', () => runDomainDetectAgent(logger)).catch(() => {}), DOMAIN_INTERVAL_MS);
  setTimeout(() => runAgent('domain', () => runDomainDetectAgent(logger)).catch(() => {}), 30_000);
  // Phase 5：关联Agent（6h 周期 + 启动后 60s 首次触发，在域检测之后）
  setInterval(() => runAgent('associate', () => runAssociateAgent(logger)).catch(() => {}), ASSOCIATE_INTERVAL_MS);
  setTimeout(() => runAgent('associate', () => runAssociateAgent(logger)).catch(() => {}), 60_000);
  // Phase 6：合成Agent（12h 周期 + 启动后 90s 首次触发）
  setInterval(() => runAgent('synthesize', () => runSynthesizeAgent(logger)).catch(() => {}), SYNTHESIZE_INTERVAL_MS);
  setTimeout(() => runAgent('synthesize', () => runSynthesizeAgent(logger)).catch(() => {}), 90_000);
  // Phase 7：Meta-Agent（24h 周期 + 启动后 120s 首次触发）
  setInterval(() => runAgent('meta', () => runMetaAgent(logger)).catch(() => {}), META_INTERVAL_MS);
  setTimeout(() => runAgent('meta', () => runMetaAgent(logger)).catch(() => {}), 120_000);

  // Obsidian Bridge：每 6 小时分层导出 + Git push（Phase 9）
  if (OBSIDIAN_VAULT) {
    logger?.info?.(`[atlas-memory] Obsidian Bridge 启动，vault: ${OBSIDIAN_VAULT}`);
    setInterval(() => runLayeredExport(logger).catch(() => {}), MIRROR_EXPORT_INTERVAL);
    setTimeout(() => runLayeredExport(logger).catch(() => {}), 15_000);
  }
  // Phase 8.5：L1知识补全Agent（8h周期 + 启动后 150s 首次触发）
  setInterval(() => runAgent('completion', () => runL1CompletionAgent(logger)).catch(() => {}), COMPLETION_INTERVAL_MS);
  setTimeout(() => runAgent('completion', () => runL1CompletionAgent(logger)).catch(() => {}), 150_000);
  // Phase 12：结构重组Agent（7天周期，不主动触发首次，等数据积累）
  setInterval(() => runAgent('restructure', () => runRestructureAgent(logger)).catch(() => {}), RESTRUCTURE_INTERVAL_MS);

  // v11 TTL 过期扫描：每小时一次，trading 直接删，其余归档
  setInterval(async () => {
    try {
      const now = new Date().toISOString();
      let offset = null;
      const expired = [];
      do {
        const body = {
          limit: 200, with_payload: true, with_vector: false,
          filter: {
            must: [{ key: 'expires_at', range: { lt: now } }],
            must_not: [
              { key: 'status', match: { value: 'superseded' } },
              { key: 'status', match: { value: 'archived' } },
            ],
          },
        };
        if (offset != null) body.offset = offset;
        const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', body);
        if (!r.ok) break;
        expired.push(...(r.body?.result?.points ?? []));
        offset = r.body?.result?.next_page_offset ?? null;
      } while (offset != null);

      if (expired.length === 0) return;

      const toDelete  = expired.filter(p => p.payload?.source_type === 'trading').map(p => p.id);
      const toArchive = expired.filter(p => p.payload?.source_type !== 'trading').map(p => p.id);

      if (toDelete.length > 0)
        await httpReq(`${QDRANT}/collections/${COLLECTION}/points/delete?wait=true`, 'POST', { points: toDelete });
      if (toArchive.length > 0)
        await httpReq(`${QDRANT}/collections/${COLLECTION}/points/payload`, 'POST', {
          payload: { status: 'archived', archived_at: now }, points: toArchive,
        });

      appendEvolutionLog('TTL_EXPIRE', `过期处理：删除${toDelete.length}条(trading)，归档${toArchive.length}条`).catch(() => {});
      logger?.info?.(`[atlas-memory] TTL过期：删除${toDelete.length}，归档${toArchive.length}`);
    } catch (e) {
      logger?.debug?.(`[atlas-memory] TTL扫描错误: ${e.message}`);
    }
  }, 60 * 60 * 1000);

  // 事件驱动：L0积压检测（每5分钟，积压≥10条立即触发整理Agent）
  setInterval(async () => {
    if (agentLocks.get('organize')) return;
    try {
      const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/count`, 'POST', {
        filter: { must: [{ key: 'level', match: { value: LEVEL_RAW } }] },
      });
      const count = r.body?.result?.count ?? 0;
      if (count >= 10) {
        logger?.info?.(`[atlas-memory] L0积压${count}条，事件触发整理Agent`);
        runAgent('organize', () => runOrganizeAgent(logger)).catch(() => {});
      }
    } catch {}
  }, 5 * 60 * 1000);

  // ══════════════════════════════════════════════════════════════════════════════
  // INJECT
  // ══════════════════════════════════════════════════════════════════════════════
  api.on('before_prompt_build', async (event, _ctx) => {
    const query = (event.prompt ?? '').trim();
    if (query.length < 15) return;
    const key = query.slice(0, 200);
    if (key === lastInjectKey) return lastInjectResult;
    try {
      // 硬超时：INJECT_TIMEOUT_MS 内未完成即降级，不阻塞响应
      const work = (async () => {
        const vector = await embed(query);
        if (!vector) return undefined;
        // v10：层级优先搜索
        const hits = await qdrantSearchForInject(vector);
        if (!hits.length) return undefined;
        const decayed = applyTimeDecay(hits, 'relevant');
        lastInjectedIds = decayed.map(h => h.id);
        trackAccess(decayed).catch(() => {});

        // 在剩余预算内读 L2/L3 源文件（每文件 500ms 上限，并行）
        const fileContents = {};
        const highLevelHits = decayed.filter(h => (h.payload?.level ?? 0) >= LEVEL_INSIGHT);
        if (highLevelHits.length && OBSIDIAN_VAULT) {
          const reads = highLevelHits.map(async h => {
            const text = await tryReadObsidianFile(h.payload?.obsidian_path, 500);
            if (text) fileContents[h.id] = text;
          });
          await Promise.allSettled(reads);
        }

        return { prependContext: formatInjectContext(decayed, fileContents) };
      })();
      const deadline = new Promise(res => setTimeout(() => res(undefined), INJECT_TIMEOUT_MS));
      const result   = await Promise.race([work, deadline]);
      lastInjectKey    = key;
      lastInjectResult = result;
      return result;
    } catch { /* 静默降级 */ }
  }, { priority: 50 });

  // ══════════════════════════════════════════════════════════════════════════════
  // CAPTURE — ★ 含用户上下文 + 质量过滤 + 冲突检测
  // ══════════════════════════════════════════════════════════════════════════════
  api.on('agent_end', (event, ctx) => {
    if (!event.success) return;
    (async () => {
      try {
        const { assistantText, userContext } = extractConversationContext(event.messages);
        if (assistantText.length < MIN_CAPTURE_CHARS) return;
        const facts = await extractFacts(assistantText, userContext);
        const valid = facts.filter(f => f.content?.trim());
        if (valid.length) {
          const r = await batchStoreMemories(valid, 'auto-capture', ctx.sessionKey, true); // ★ 冲突检测开启
          if (r.stored > 0) logger?.debug?.(`[atlas-memory] agent_end: +${r.stored} 条，去重${r.deduplicated}，冲突跳过${r.skipped}`);
        }
      } catch (e) {
        logger?.debug?.(`[atlas-memory] capture error: ${e.message}`);
      }
    })();
  });

  // ══════════════════════════════════════════════════════════════════════════════
  // CAPTURE — 中途捕获（每 N 轮）
  // ══════════════════════════════════════════════════════════════════════════════
  api.on('llm_output', (event, ctx) => {
    const sid   = event.sessionId ?? ctx.sessionId ?? 'unknown';
    const count = (sessionTurns.get(sid) ?? 0) + 1;
    sessionTurns.set(sid, count);
    if (count % CAPTURE_TURN_INTERVAL !== 0) return;
    const text = (event.assistantTexts ?? []).join('\n\n').trim();
    if (text.length < MIN_CAPTURE_CHARS) return;
    (async () => {
      try {
        const facts = await extractFacts(text);
        const valid = facts.filter(f => f.content?.trim());
        if (valid.length) {
          const r = await batchStoreMemories(valid, 'mid-capture', ctx.sessionKey, false);
          if (r.stored > 0) logger?.debug?.(`[atlas-memory] mid-capture: +${r.stored} 条（第${count}轮）`);
        }
      } catch (e) {
        logger?.debug?.(`[atlas-memory] mid-capture error: ${e.message}`);
      }
    })();
  });

  api.on('session_end', (_event, ctx) => {
    if (ctx.sessionId) sessionTurns.delete(ctx.sessionId);
  });

  // ══════════════════════════════════════════════════════════════════════════════
  // LEARN — 搜索工具自动学习
  // ══════════════════════════════════════════════════════════════════════════════
  api.on('after_tool_call', (event, ctx) => {
    if (event.error || !isSearchTool(event.toolName)) return;
    (async () => {
      try {
        const query = event.params?.query ?? event.params?.q ?? event.params?.keyword ?? event.toolName;
        const text  = searchResultToText(event.result, query);
        if (text.length < 80) return;
        const facts = await extractWebFacts(text, String(query));
        const valid = facts.filter(f => f.content?.trim());
        if (valid.length) {
          const r = await batchStoreMemories(valid, `web-learn:${event.toolName}`, ctx.sessionKey, false);
          if (r.stored > 0) logger?.info?.(`[atlas-memory] web-learn: ${event.toolName} → +${r.stored} 条知识`);
        }
      } catch (e) {
        logger?.debug?.(`[atlas-memory] web-learn error (${event.toolName}): ${e.message}`);
      }
    })();
  });

  // ══════════════════════════════════════════════════════════════════════════════
  // SUPPLEMENT
  // ══════════════════════════════════════════════════════════════════════════════
  api.registerMemoryCorpusSupplement({
    async search({ query, maxResults = 5 }) {
      try {
        const vector  = await embed(query);
        if (!vector) return [];
        const hits    = await qdrantSearch(vector, { limit: maxResults, minScore: 0.50 });
        const decayed = applyTimeDecay(hits, 'relevant');
        return decayed.map(h => ({
          corpus:     'atlas',
          path:       `atlas:${h.id}`,
          title:      h.payload?.tags?.[0] ?? h.payload?.category ?? 'memory',
          kind:       'atlas-memory',
          score:      h.effectiveScore ?? h.score,
          snippet:    (h.payload?.content ?? '').slice(0, 300),
          source:     'atlas-memory',
          sourceType: 'vector-db',
          updatedAt:  h.payload?.created_at,
        }));
      } catch { return []; }
    },
    async get() { return null; },
  });

  // ══════════════════════════════════════════════════════════════════════════════
  // TOOLS
  // ══════════════════════════════════════════════════════════════════════════════

  // atlas_intake（v12 新增：统一录入入口）
  api.registerTool(() => ({
    name: 'atlas_intake',
    description: '统一知识录入入口（v12）。接受任意内容，自动检测分片信号，立即写入L0，后台整理Agent异步提炼L1。支持多片段通过group_id关联。',
    parameters: {
      type: 'object', required: ['content'],
      properties: {
        content:      { type: 'string', description: '原始内容（任意长度）' },
        content_type: { type: 'string', enum: ['video_script','article','note','chat_log','trading_signal','sop','course','news','social_post','other'], description: '内容类型，影响整理策略' },
        group_id:     { type: 'string', description: '分片组ID，同一来源多片段共享同一ID' },
        group_total:  { type: 'number', description: '该组总片段数' },
        chunk_index:  { type: 'number', description: '当前片段序号（0开始）' },
        source_meta:  { type: 'object', description: '来源元数据，如 {platform, url, author, recorded_at}' },
        domain:       { type: 'string', description: '知识域（可选，自动推断）' },
        topic:        { type: 'string', description: '主题标题（可选）' },
        tags:         { type: 'array',  items: { type: 'string' } },
        importance:   { type: 'string', enum: ['low','medium','high','critical'] },
      },
    },
    execute: async (_callId, params) => {
      const { content, content_type = 'other', group_id, group_total, chunk_index,
        source_meta, domain, topic, tags = [], importance = 'medium' } = params ?? {};
      if (!content?.trim()) return jsonResult({ error: 'content 不能为空' });
      const r = await intakeToL0({ content, content_type, group_id, group_total, chunk_index,
        source_meta, domain, topic, tags: Array.isArray(tags) ? tags : [], importance, source: 'plugin_intake' });
      if (!r.ok) return jsonResult({ error: r.error ?? 'intake 失败' });
      return jsonResult({ ok: true, id: r.id, group_id: r.group_id ?? null,
        is_fragment: !!(r.group_id), content_type });
    },
  }));

  // atlas_store（v12：自由payload + 新字段）
  api.registerTool(() => ({
    name: 'atlas_store',
    description: '直接存入知识库（v12）。自动检测 source_type 和 TTL。支持任意额外字段，适合Agent已处理好的结构化内容。',
    parameters: {
      type: 'object', required: ['content'],
      additionalProperties: true,
      properties: {
        content:      { type: 'string',  description: '要存储的内容' },
        importance:   { type: 'string',  enum: ['low', 'medium', 'high', 'critical'], default: 'medium' },
        memory_type:  { type: 'string',  default: 'fact' },
        tags:         { type: 'array',   items: { type: 'string' }, default: [] },
        topic:        { type: 'string' },
        domain:       { type: 'string' },
        knowledge_type: { type: 'string' },
        content_type: { type: 'string' },
        group_id:     { type: 'string' },
        source_meta:  { type: 'object' },
        platform:     { type: 'string' },
      },
    },
    execute: async (_callId, params) => {
      const { content, importance = 'medium', memory_type = 'fact', tags = [], topic, domain,
        knowledge_type = 'capture', content_type, group_id, source_meta, platform } = params ?? {};
      if (!content?.trim()) return jsonResult({ error: 'content 不能为空' });
      const detectedSourceType = detectSourceType(content, '', Array.isArray(tags) ? tags : []);
      const r = await intakeToL0({ content, importance, memory_type, tags, topic, domain,
        source: 'plugin', knowledge_type, content_type, group_id,
        source_meta: source_meta ?? (platform ? { platform } : null) });
      if (!r.ok) return jsonResult({ error: r.error });
      return jsonResult({ ok: true, id: r.id, source_type: detectedSourceType,
        ttl: TTL_MAP[detectedSourceType] ?? 'permanent' });
    },
  }));

  // atlas_recall（v12：实体扩展 + 置信度过滤 + intent）
  api.registerTool(() => ({
    name: 'atlas_recall',
    description: '从知识库检索相关知识（v12：实体扩展 + 置信度过滤 + 意图感知 + TTL自动过滤）。',
    parameters: {
      type: 'object', required: ['query'],
      properties: {
        query:           { type: 'string',  description: '搜索查询（自然语言）' },
        limit:           { type: 'integer', default: 5, minimum: 1, maximum: 20 },
        category:        { type: 'string',  enum: ['personal', 'work', 'project', 'system', 'learning', 'any'], default: 'any' },
        min_score:       { type: 'number',  default: 0.65, minimum: 0.1, maximum: 1.0 },
        source_type:     { type: 'string',  description: '按信息源类型过滤' },
        platform:        { type: 'string',  description: '按平台过滤' },
        intent:          { type: 'string',  enum: ['relevant', 'latest'], default: 'relevant', description: 'relevant=语义相关, latest=时间最新' },
        min_confidence:  { type: 'number',  default: 0, minimum: 0, maximum: 1, description: '最低置信度（0=不过滤）' },
        expand_entities: { type: 'boolean', default: false, description: '基于实体扩展搜索范围' },
      },
    },
    execute: async (_callId, params) => {
      const { query, limit = 5, category = 'any', min_score = SCORE_MIN,
        source_type, platform, intent = 'relevant', min_confidence = 0, expand_entities = false } = params ?? {};
      if (!query?.trim()) return jsonResult({ error: 'query 不能为空', results: [], count: 0 });
      const vector = await embed(query);
      if (!vector) return jsonResult({ error: 'embed 不可用', results: [], count: 0 });
      const hits = await qdrantSearch(vector, { limit: expand_entities ? limit * 2 : limit,
        category, minScore: min_score, source_type, platform, intent, min_confidence, expand_entities });
      const sorted = applyTimeDecay(hits, intent).slice(0, limit);
      trackAccess(sorted).catch(() => {});
      return jsonResult({ query, count: sorted.length, intent, results: fmtHits(sorted) });
    },
  }));

  // atlas_delete
  api.registerTool(() => ({
    name: 'atlas_delete',
    description: '按语义相似度删除记忆（建议先用 atlas_recall 确认）。',
    parameters: {
      type: 'object', required: ['query'],
      properties: {
        query:     { type: 'string', description: '要删除内容的描述' },
        min_score: { type: 'number', default: 0.85 },
      },
    },
    execute: async (_callId, params) => {
      const { query, min_score = 0.85 } = params ?? {};
      if (!query?.trim()) return jsonResult({ error: 'query 不能为空' });
      const vector = await embed(query);
      if (!vector) return jsonResult({ error: 'embed 失败' });
      const hits = await qdrantSearch(vector, { limit: 10, minScore: min_score });
      if (!hits.length) return jsonResult({ deleted: 0, message: '未找到匹配记忆' });
      const ids = hits.map(h => h.id);
      const r   = await qdrantDelete(ids);
      return jsonResult(r.ok ? { deleted: r.deleted, ids } : { error: 'Qdrant 删除失败' });
    },
  }));

  // atlas_stats
  api.registerTool(() => ({
    name: 'atlas_stats',
    description: '查看 ATLAS 记忆库完整状态：记忆数、模型、缓存命中率、备份时间。',
    parameters: { type: 'object', properties: {} },
    execute: async () => {
      const [qdrantRes, ollamaRes, omlxRes, levelStats] = await Promise.all([
        httpReq(`${QDRANT}/collections/${COLLECTION}`),
        httpReq(`${OLLAMA}/api/tags`),
        httpReq(`${OMLX}/v1/models`),
        getLevelStats(),
      ]);
      const ollamaModels = ollamaRes.ok ? (ollamaRes.body?.models ?? []).map(m => m.name) : [];
      const omlxModels   = omlxRes.ok  ? (omlxRes.body?.data   ?? []).map(m => m.id)    : [];
      const total        = embedCacheHits + embedCacheMisses;
      return jsonResult({
        version: '11.0.0',
        qdrant: {
          ok:         qdrantRes.ok,
          collection: COLLECTION,
          points:     qdrantRes.body?.result?.points_count ?? 0,
          levels:     { L0: levelStats[0], L1: levelStats[1], L2: levelStats[2], L3: levelStats[3] },
        },
        models: {
          embed:   { service: 'Ollama', model: EMBED_MODEL, ok: ollamaRes.ok, available: ollamaModels },
          extract: { service: 'omlx',   model: OMLX_MODEL,  ok: omlxRes.ok,  available: omlxModels },
        },
        embed_cache: {
          size: embedCache.size, capacity: EMBED_CACHE_SIZE,
          hits: embedCacheHits, misses: embedCacheMisses,
          hit_rate: total ? `${Math.round(embedCacheHits / total * 100)}%` : 'n/a',
        },
        quality_threshold: MIN_QUALITY_SCORE,
        stale_pruning:     `hit_count=0 + age>${STALE_AGE_DAYS}天 + importance=low`,
        backup:            { dir: BACKUP_DIR, last_backup: lastBackupTime ?? '未备份' },
        active_sessions:   sessionTurns.size,
        features: {
          extract_model:   `omlx ${OMLX_MODEL}（4s/次，thinking关闭）`,
          distill_model:   `${DEEPSEEK_MODEL}（云端，omlx备用）`,
          conflict_detect: `开启（agent_end+atlas_store，medium+重要性触发）`,
          quality_filter:  `≥${MIN_QUALITY_SCORE}/10`,
          memory_types:    'preference|fact|skill|project|constraint|event|[distilled]',
          time_decay:      `${DECAY_PERIOD_DAYS}天半衰期，最大惩罚${DECAY_MAX_PENALTY * 100}%`,
          feedback:        `正反馈+${FEEDBACK_BOOST}，负反馈-${FEEDBACK_DECAY}，删除门槛${FEEDBACK_DELETE_FLOOR}`,
          versioning:      '冲突替换保留历史（status:superseded），不物理删除',
          auto_distill:    `EVOLVE 24h 扫描，同标签≥${DISTILL_MIN_COUNT}条自动提炼通则`,
        },
        obsidian_bridge: {
          enabled:     Boolean(OBSIDIAN_VAULT),
          vault:       OBSIDIAN_VAULT || '未配置（设置 ATLAS_OBSIDIAN_VAULT 环境变量）',
          mirror_dir:  OBSIDIAN_VAULT ? join(OBSIDIAN_VAULT, OBSIDIAN_MIRROR_DIR) : null,
          export_interval: `每 ${MIRROR_EXPORT_INTERVAL / 3600000}h 自动刷新`,
          features:    '主题聚类导出 · 每日进化日志 · Dataview 仪表盘 · 图谱 wikilinks',
        },
      });
    },
  }));

  // atlas_evolve
  api.registerTool(() => ({
    name: 'atlas_evolve',
    description: '手动触发记忆进化：相似度去重（≥0.92）+ 过期记忆清理（无访问+90天+低重要性）。',
    parameters: { type: 'object', properties: {} },
    execute: async () => {
      try {
        const result = await runEvolution(logger);
        return jsonResult({ ok: true, ...result });
      } catch (e) {
        return jsonResult({ error: e.message });
      }
    },
  }));

  // atlas_organize（Phase 3）
  api.registerTool(() => ({
    name: 'atlas_organize',
    description: '手动触发整理Agent：将L0原料晋升为L1知识，自动域归类，写入Obsidian L1文件。每次最多处理20条。',
    parameters: { type: 'object', properties: {} },
    execute: async () => {
      try {
        if (agentLocks.get('organize')) return jsonResult({ ok: false, reason: '整理Agent正在运行，请稍后重试' });
        const result = await new Promise((resolve, reject) => {
          runAgent('organize', () => runOrganizeAgent(logger)).then(resolve).catch(reject);
        });
        return jsonResult({ ok: true, ...(result ?? {}) });
      } catch (e) {
        return jsonResult({ error: e.message });
      }
    },
  }));

  // atlas_domain_detect（Phase 4）
  api.registerTool(() => ({
    name: 'atlas_domain_detect',
    description: '手动触发域检测Agent：对domain=null的未分类记录做向量聚类，自动推断新域名（DeepSeek），创建Obsidian目录+维度图谱，更新Qdrant域字段。',
    parameters: { type: 'object', properties: {} },
    execute: async () => {
      try {
        if (agentLocks.get('domain')) return jsonResult({ ok: false, reason: '域检测Agent正在运行，请稍后重试' });
        const result = await new Promise((resolve, reject) => {
          runAgent('domain', () => runDomainDetectAgent(logger)).then(resolve).catch(reject);
        });
        return jsonResult({ ok: true, ...(result ?? {}) });
      } catch (e) {
        return jsonResult({ error: e.message });
      }
    },
  }));

  // atlas_associate（Phase 5）
  api.registerTool(() => ({
    name: 'atlas_associate',
    description: '手动触发关联Agent：扫描新增L1节点，跨域碰撞生成L2洞见，写入两个域的L2-关联/目录，追加wikilinks到L1源文件。',
    parameters: { type: 'object', properties: {} },
    execute: async () => {
      try {
        if (agentLocks.get('associate')) return jsonResult({ ok: false, reason: '关联Agent正在运行，请稍后重试' });
        const result = await new Promise((resolve, reject) => {
          runAgent('associate', () => runAssociateAgent(logger)).then(resolve).catch(reject);
        });
        return jsonResult({ ok: true, ...(result ?? {}) });
      } catch (e) {
        return jsonResult({ error: e.message });
      }
    },
  }));

  // atlas_synthesize（Phase 6）
  api.registerTool(() => ({
    name: 'atlas_synthesize',
    description: '手动触发合成Agent：扫描L2洞见聚类（≥3），用DeepSeek合成L3智识框架，版本化写入Obsidian L3-智识/目录，更新L2节点derived_to_id。',
    parameters: { type: 'object', properties: {} },
    execute: async () => {
      try {
        if (agentLocks.get('synthesize')) return jsonResult({ ok: false, reason: '合成Agent正在运行，请稍后重试' });
        const result = await new Promise((resolve, reject) => {
          runAgent('synthesize', () => runSynthesizeAgent(logger)).then(resolve).catch(reject);
        });
        return jsonResult({ ok: true, ...(result ?? {}) });
      } catch (e) {
        return jsonResult({ error: e.message });
      }
    },
  }));

  // atlas_meta（Phase 7）
  api.registerTool(() => ({
    name: 'atlas_meta',
    description: '手动触发Meta-Agent：批量更新新鲜度衰减、统计各域知识分布、DeepSeek生成采集计划、执行web搜索补充L0、写日报到Obsidian _系统/日报/。',
    parameters: { type: 'object', properties: {} },
    execute: async () => {
      try {
        if (agentLocks.get('meta')) return jsonResult({ ok: false, reason: 'Meta-Agent正在运行，请稍后重试' });
        const result = await new Promise((resolve, reject) => {
          runAgent('meta', () => runMetaAgent(logger)).then(resolve).catch(reject);
        });
        return jsonResult({ ok: true, ...(result ?? {}) });
      } catch (e) {
        return jsonResult({ error: e.message });
      }
    },
  }));

  // atlas_complete_gaps（L1知识补全）
  api.registerTool(() => ({
    name: 'atlas_complete_gaps',
    description: '手动触发L1知识补全Agent：扫描完整度低于85%的L1节点，用searcharvester搜索+DeepSeek补全缺失字段，写回Qdrant和Obsidian vault。每次最多处理5个节点、每节点最多补3个字段。',
    parameters: { type: 'object', properties: {} },
    execute: async () => {
      try {
        if (agentLocks.get('completion')) return jsonResult({ ok: false, reason: 'L1补全Agent正在运行，请稍后重试' });
        const result = await new Promise((resolve, reject) => {
          runAgent('completion', () => runL1CompletionAgent(logger)).then(resolve).catch(reject);
        });
        return jsonResult({ ok: true, ...(result ?? {}) });
      } catch (e) {
        return jsonResult({ error: e.message });
      }
    },
  }));

  // atlas_web_learn（分块学习）
  api.registerTool(() => ({
    name: 'atlas_web_learn',
    description: '从 URL 或文本中学习并提取知识存入记忆库。长文章自动分块（≤5块×1500字）。',
    parameters: {
      type: 'object', required: [],
      properties: {
        url:       { type: 'string',  description: '网页 URL（http/https）' },
        text:      { type: 'string',  description: '直接提供的文本内容（与 url 二选一）' },
        query:     { type: 'string',  description: '主题方向描述（可选）' },
        max_facts: { type: 'integer', default: 5, minimum: 1, maximum: 10 },
      },
    },
    execute: async (_callId, params) => {
      const { url, text: rawText, query = '', max_facts = 5 } = params ?? {};
      let content = '';
      if (url) {
        const fetched = await fetchUrlText(url);
        if (!fetched.ok) return jsonResult({ error: `抓取失败: ${fetched.error ?? '未知'}`, url });
        content = fetched.contentType?.includes('html') ? htmlToText(fetched.text) : fetched.text;
        if (content.length < 100) return jsonResult({ error: '页面内容过短', url });
      } else if (rawText) {
        content = rawText;
      } else {
        return jsonResult({ error: '请提供 url 或 text 参数' });
      }

      const chunks   = chunkText(content).slice(0, MAX_CHUNKS);
      const allFacts = [];
      const sys      = '你是知识提取助手。严格只输出有效JSON数组，不要解释。';
      for (let i = 0; i < chunks.length; i++) {
        const chunkQuery = query || url || `内容 ${i + 1}/${chunks.length}`;
        const user =
          `从以下网页内容提取最多 ${max_facts} 条有价值的知识（质量<7不提取）。
主题：${chunkQuery.slice(0, 100)}
格式：[{"content":"...","category":"learning|work|project|system","importance":"low|medium|high|critical","tags":[],"quality":7,"memory_type":"fact|skill|constraint"}]

内容（${i + 1}/${chunks.length}）：
${chunks[i]}

JSON数组：`;
        const out = await omlxGenerate(sys, user, 1000);
        if (out) {
          const facts = parseFactsJson(out);
          allFacts.push(...facts.filter(f => (f.quality ?? 10) >= MIN_QUALITY_SCORE));
        }
      }

      if (!allFacts.length) return jsonResult({ ok: true, stored: 0, message: '未提取到高质量知识', chunks: chunks.length });

      const hostname = url ? (() => { try { return new URL(url).hostname; } catch { return ''; } })() : '';
      const facts    = allFacts.map(f => ({ ...f, tags: [...(f.tags ?? []), ...(hostname ? [hostname] : [])] }));
      const r        = await batchStoreMemories(facts, url ? `web-learn:${url}` : 'web-learn:manual', undefined, false);
      return jsonResult({ ok: true, url: url ?? null, chunks: chunks.length, extracted: allFacts.length, stored: r.stored, deduplicated: r.deduplicated });
    },
  }));

  // ★ atlas_merge — 近重复智能合并
  api.registerTool(() => ({
    name: 'atlas_merge',
    description:
      '扫描记忆库中相似度在 0.75-0.92 的近重复条目，用 Qwen3.5 合并成更丰富的单条记忆。' +
      '提升记忆质量，减少冗余碎片。',
    parameters: {
      type: 'object', properties: {
        query:      { type: 'string',  description: '指定合并主题范围（可选，默认全库）' },
        min_score:  { type: 'number',  default: 0.78, minimum: 0.70, maximum: 0.91 },
        max_merges: { type: 'integer', default: 10,   minimum: 1,    maximum: 50  },
      },
    },
    execute: async (_callId, params) => {
      const { query, min_score = 0.78, max_merges = 10 } = params ?? {};

      // 获取候选点
      let candidates = [];
      if (query) {
        const vector = await embed(query);
        if (!vector) return jsonResult({ error: 'embed 失败' });
        candidates = await qdrantSearch(vector, { limit: 100, minScore: 0.65 });
      } else {
        let offset = null;
        do {
          const body = { limit: 250, with_payload: true, with_vector: true };
          if (offset != null) body.offset = offset;
          const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', body);
          if (!r.ok) break;
          candidates.push(...(r.body?.result?.points ?? []).map(p => ({ ...p, score: 1.0 })));
          offset = r.body?.result?.next_page_offset ?? null;
        } while (offset != null);
      }

      const processed = new Set();
      const mergeOps  = [];

      for (const pt of candidates) {
        if (processed.has(pt.id) || !pt.vector) continue;
        const similar   = await qdrantSearch(pt.vector, { limit: 5, minScore: min_score });
        const toMerge   = similar.filter(s => s.id !== pt.id && s.score < SCORE_DEDUP && !processed.has(s.id));
        if (!toMerge.length) continue;

        const all     = [pt, ...toMerge];
        const sys     = '你是记忆合并专家。将多条相关记忆合并为一条更完整的记忆。严格只输出JSON对象。';
        const content = all.map((m, i) => `${i + 1}. ${m.payload?.content ?? ''}`).join('\n');
        const user    =
          `将以下相关记忆合并为一条更丰富、更完整的记忆：
${content}

输出JSON：{"content":"合并后的完整内容（保留所有关键信息）","category":"...","importance":"...","tags":["..."]}`;
        const out = await omlxGenerate(sys, user, 400);
        if (!out) continue;
        const merged = parseJsonObject(out);
        if (!merged?.content) continue;

        all.forEach(m => processed.add(m.id));
        mergeOps.push({ ids: all.map(m => m.id), merged });
        if (mergeOps.length >= max_merges) break;
      }

      // 执行合并
      let mergedCount  = 0;
      let idsRemoved   = 0;
      for (const op of mergeOps) {
        await qdrantDelete(op.ids);
        idsRemoved += op.ids.length;
        const vec = await embed(op.merged.content);
        if (!vec) continue;
        const now = new Date().toISOString();
        await httpReq(`${QDRANT}/collections/${COLLECTION}/points?wait=true`, 'PUT', {
          points: [{
            id:      stableId(op.merged.content + now),
            vector:  vec,
            payload: {
              ...op.merged,
              tags:             Array.isArray(op.merged.tags) ? op.merged.tags : [],
              memory_type:      op.merged.memory_type ?? 'fact',
              created_at:       now,
              source:           'atlas_merge',
              hit_count:        0,
              last_accessed_at: null,
            },
          }],
        });
        mergedCount++;
      }

      if (mergedCount > 0) {
        appendEvolutionLog('MERGE', `合并 ${mergedCount} 组 → 清理 ${idsRemoved} 条，净减少 ${idsRemoved - mergedCount} 条`).catch(() => {});
      }
      return jsonResult({ ok: true, merged_groups: mergedCount, ids_removed: idsRemoved, net_reduction: idsRemoved - mergedCount });
    },
  }));

  // atlas_export
  api.registerTool(() => ({
    name: 'atlas_export',
    description: '将记忆库导出为 JSON 文件（含向量），用于备份或迁移。默认保存至 ~/.atlas-backups/。',
    parameters: { type: 'object', properties: { path: { type: 'string', description: '自定义导出路径（可选）' } } },
    execute: async (_callId, params) => {
      try { return jsonResult(await backupCollection(logger, params?.path)); }
      catch (e) { return jsonResult({ error: e.message }); }
    },
  }));

  // atlas_import
  api.registerTool(() => ({
    name: 'atlas_import',
    description: '从 JSON 备份文件恢复记忆库（atlas_export 格式，合并模式导入）。',
    parameters: { type: 'object', required: ['path'], properties: { path: { type: 'string' } } },
    execute: async (_callId, params) => {
      const { path: filePath } = params ?? {};
      if (!filePath) return jsonResult({ error: '请提供 path 参数' });
      try {
        const raw  = await readFile(filePath, 'utf8');
        const data = JSON.parse(raw);
        const pts  = data.points ?? [];
        if (!pts.length) return jsonResult({ ok: true, imported: 0, total: 0 });
        await ensureCollection();
        let imported = 0;
        for (let i = 0; i < pts.length; i += 100) {
          const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points?wait=true`, 'PUT', { points: pts.slice(i, i + 100) });
          if (r.ok) imported += Math.min(100, pts.length - i);
        }
        return jsonResult({ ok: true, imported, total: pts.length, file: filePath });
      } catch (e) {
        return jsonResult({ error: e.message });
      }
    },
  }));

  // atlas_obsidian_sync — Phase 9：分层导出 + Git push
  api.registerTool(() => ({
    name: 'atlas_obsidian_sync',
    description:
      '手动触发 Obsidian 分层导出（Phase 9）：①各域 _index.md（Dataview）；' +
      '②_系统/域图谱.md 全局汇总；③Git commit+push 到 knowledge-base 仓库。',
    parameters: { type: 'object', properties: {} },
    execute: async () => {
      if (!OBSIDIAN_VAULT) return jsonResult({ error: 'ATLAS_OBSIDIAN_VAULT 未配置' });
      try {
        const result = await runLayeredExport(logger);
        await appendEvolutionLog('SYNC', `分层导出: ${result.domains}域, git=${result.pushed}`);
        return jsonResult({ ok: true, vault: OBSIDIAN_VAULT, ...result });
      } catch (e) {
        return jsonResult({ error: e.message });
      }
    },
  }));

  // ★ atlas_feedback — 记忆反馈回路
  api.registerTool(() => ({
    name: 'atlas_feedback',
    description:
      '对刚才引用的记忆进行反馈评价。' +
      'correct=提升权重，wrong/outdated=降低权重（累计低于阈值自动删除）。' +
      '用户说"不对/你记错了/过时了"时主动调用。',
    parameters: {
      type: 'object', required: ['verdict'],
      properties: {
        query:   { type: 'string',  description: '记忆内容的关键词（用于语义定位，与 id 二选一）' },
        id:      { type: 'string',  description: '记忆 ID（从 atlas_recall 结果获取，优先使用）' },
        verdict: { type: 'string',  enum: ['correct', 'wrong', 'outdated'], description: '评价结果' },
        reason:  { type: 'string',  description: '评价原因（可选）' },
      },
    },
    execute: async (_callId, params) => {
      const { query, id, verdict, reason = '' } = params ?? {};
      if (!verdict) return jsonResult({ error: 'verdict 不能为空' });

      let targetId      = id ?? null;
      let targetContent = '';

      // 未提供 id：语义搜索，优先从最近注入的记忆中匹配
      if (!targetId && query) {
        const vector = await embed(query);
        if (vector) {
          const hits = await qdrantSearch(vector, { limit: 5, minScore: 0.6 });
          const recentHit = hits.find(h => lastInjectedIds.includes(h.id)) ?? hits[0];
          if (recentHit) { targetId = recentHit.id; targetContent = recentHit.payload?.content?.slice(0, 80) ?? ''; }
        }
      }
      if (!targetId) return jsonResult({ error: '未找到目标记忆，请提供 id 或更精确的 query' });

      // 获取当前 feedback_score
      const getR = await httpReq(`${QDRANT}/collections/${COLLECTION}/points`, 'POST', {
        ids: [targetId], with_payload: true, with_vector: false,
      });
      if (!getR.ok || !getR.body?.result?.[0]) return jsonResult({ error: '记忆不存在或已删除' });

      const current      = getR.body.result[0];
      const currentScore = current.payload?.feedback_score ?? 1.0;
      targetContent      = targetContent || current.payload?.content?.slice(0, 80) || '';

      const delta    = verdict === 'correct' ? FEEDBACK_BOOST : -FEEDBACK_DECAY;
      const newScore = Math.max(0, Math.min(1, currentScore + delta));

      if (newScore <= FEEDBACK_DELETE_FLOOR) {
        await qdrantDelete([targetId]);
        appendEvolutionLog('FEEDBACK', `删除（负评累积）"${targetContent}" ${reason ? `[${reason}]` : ''}`).catch(() => {});
        return jsonResult({
          ok: true, action: 'deleted',
          reason: `feedback_score ${currentScore.toFixed(2)}→${newScore.toFixed(2)} ≤ ${FEEDBACK_DELETE_FLOOR}`,
          memory: targetContent,
        });
      }

      await qdrantPatchPayload(targetId, { feedback_score: newScore });
      appendEvolutionLog('FEEDBACK', `${verdict === 'correct' ? '✓' : '✗'} "${targetContent}" score:${currentScore.toFixed(2)}→${newScore.toFixed(2)} ${reason ? `[${reason}]` : ''}`).catch(() => {});
      return jsonResult({
        ok: true, action: 'updated', verdict,
        feedback_score: { before: currentScore, after: newScore },
        memory: targetContent,
      });
    },
  }));

  // ★ atlas_distill — 知识提炼（DeepSeek 云端合成通则）
  api.registerTool(() => ({
    name: 'atlas_distill',
    description:
      '对指定标签下的多条记忆进行知识提炼，使用 DeepSeek 合成一条高质量"通则"（不足 5 条则报错）。' +
      '通则会优先注入到下次对话上下文中。',
    parameters: {
      type: 'object', required: ['tag'],
      properties: {
        tag:   { type: 'string',  description: '要提炼的标签名' },
        force: { type: 'boolean', default: false, description: '强制重新提炼（覆盖已有通则）' },
      },
    },
    execute: async (_callId, params) => {
      const { tag, force = false } = params ?? {};
      if (!tag?.trim()) return jsonResult({ error: 'tag 不能为空' });
      const result = await distillTagMemories(tag.trim(), logger, force);
      if (!result) return jsonResult({ error: 'distill 失败（DeepSeek 和 omlx 均不可用）' });
      if (result.skipped) return jsonResult({ ok: true, skipped: true, reason: result.reason });
      appendEvolutionLog('DISTILL', `手动提炼"${tag}"：${result.basis}条 → 通则 (id:${String(result.id)?.slice(0, 8)})`).catch(() => {});
      return jsonResult({ ok: true, tag, ...result });
    },
  }));

  // ★ atlas_timeline — 主题时间线
  api.registerTool(() => ({
    name: 'atlas_timeline',
    description: '按时间线查看知识演进（创建时间排序）。tag 可选：填写则过滤该标签，不填则返回全库最新条目。',
    parameters: {
      type: 'object',
      properties: {
        tag:   { type: 'string',  description: '标签名或域名（可选，不填返回全局最新）' },
        limit: { type: 'integer', default: 20, minimum: 1, maximum: 100 },
        order: { type: 'string',  enum: ['asc', 'desc'], default: 'desc', description: 'desc=最新在前' },
      },
    },
    execute: async (_callId, params) => {
      const { tag, limit = 20, order = 'desc' } = params ?? {};
      const tagStr = tag?.trim() ?? '';

      let offset = null;
      const allPoints = [];
      do {
        const mustNot = [{ key: 'status', match: { value: 'superseded' } }];
        const must = tagStr ? [{ key: 'tags', match: { value: tagStr } }] : [];
        const body = {
          limit: 100, with_payload: true, with_vector: false,
          filter: { must, must_not: mustNot },
        };
        if (offset != null) body.offset = offset;
        const r = await httpReq(`${QDRANT}/collections/${COLLECTION}/points/scroll`, 'POST', body);
        if (!r.ok) break;
        allPoints.push(...(r.body?.result?.points ?? []));
        offset = r.body?.result?.next_page_offset ?? null;
      } while (offset != null && allPoints.length < 500);

      if (!allPoints.length) return jsonResult({ tag: tagStr || null, count: 0, timeline: [] });

      const sorted = allPoints
        .sort((a, b) => {
          const ta = new Date(a.payload?.created_at ?? 0).getTime();
          const tb = new Date(b.payload?.created_at ?? 0).getTime();
          return order === 'desc' ? tb - ta : ta - tb;
        })
        .slice(0, limit);

      return jsonResult({
        tag:      tagStr || null,
        total:    allPoints.length,
        showing:  sorted.length,
        timeline: sorted.map(h => ({
          id:           h.id,
          level:        h.payload?.level ?? 0,
          domain:       h.payload?.domain ?? null,
          topic:        h.payload?.topic ?? null,
          date:         (h.payload?.created_at ?? '').slice(0, 10),
          content:      (h.payload?.content ?? '').slice(0, 100),
          importance:   h.payload?.importance,
          memory_type:  h.payload?.memory_type,
          hit_count:    h.payload?.hit_count ?? 0,
          feedback_score: h.payload?.feedback_score ?? 1.0,
          is_distilled: (h.payload?.tags ?? []).includes(DISTILL_TAG),
        })),
      });
    },
  }));

  // atlas_restructure — Phase 12：结构重组Agent
  api.registerTool(() => ({
    name: 'atlas_restructure',
    description:
      '手动触发结构重组Agent（Phase 12）：' +
      '①检测语义高度重叠的域（质心相似度≥0.88）并合并；' +
      '②检测内聚度过低的域（<0.55）并用 DeepSeek 命名后分裂；' +
      '③更新 Qdrant + Obsidian + 运行时缓存。每次最多合并2次、分裂1次。',
    parameters: { type: 'object', properties: {} },
    execute: async () => {
      try {
        if (agentLocks.get('restructure'))
          return jsonResult({ ok: false, reason: '重组Agent正在运行，请稍后重试' });
        const result = await new Promise((resolve, reject) =>
          runAgent('restructure', () => runRestructureAgent(logger)).then(resolve).catch(reject)
        );
        return jsonResult({ ok: true, ...(result ?? {}) });
      } catch (e) {
        return jsonResult({ error: e.message });
      }
    },
  }));
}
