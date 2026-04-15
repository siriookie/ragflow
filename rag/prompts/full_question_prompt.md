## 角色
一个有帮助的助手。

## 任务与步骤
1. 根据下面的对话，补全用户接下来真正想问的完整问题。
2. 如果用户的问题里包含相对日期，请基于今天（{{ today }}）将其改写为绝对日期。
   - “昨天” = {{ yesterday }}，“明天” = {{ tomorrow }}

## 要求与限制
- 如果用户最新的问题已经足够完整，就不要改写，直接返回原问题。
- 除了改写后的问题本身，不要输出任何额外内容。
{% if language %}
- 生成的文本必须使用 {{ language }}。
{% else %}
- 生成的文本必须与原始用户问题保持相同语言。
{% endif %}

---

## 示例

### 示例 1
**对话：**

USER: What is the name of Donald Trump's father?
ASSISTANT: Fred Trump.
USER: And his mother?

**输出：** What's the name of Donald Trump's mother?

---

### 示例 2
**对话：**

USER: What is the name of Donald Trump's father?
ASSISTANT: Fred Trump.
USER: And his mother?
ASSISTANT: Mary Trump.
USER: What's her full name?

**输出：** What's the full name of Donald Trump's mother Mary Trump?

---

### 示例 3
**对话：**

USER: What's the weather today in London?
ASSISTANT: Cloudy.
USER: What's about tomorrow in Rochester?

**输出：** What's the weather in Rochester on {{ tomorrow }}?

---

## 实际数据

**对话：**

{{ conversation }}
