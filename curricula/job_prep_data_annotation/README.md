# Job Prep: DataAnnotation Coding Assessment

**Curriculum Type:** LINEAR  
**Duration:** ~3 hours  
**Target:** Candidates preparing for DataAnnotation coding assessments

---

## 🎯 What This Curriculum Teaches

This curriculum provides **surgical training** for the specific Python skills required to pass DataAnnotation's practical coding section. It was designed through forensic analysis of actual assessment failures.

### The Problem This Solves

Many candidates have strong **algorithmic intuition** but fail because they attempt to apply **generic pseudocode** to **specific Python syntax**. This curriculum bridges that gap.

**Documented Failure Pattern:**
- ✅ Passed: Logic/QA sections (algorithm design, bug identification)
- ❌ Failed: Coding section (HTTP requests, HTML parsing, 2D grids)
- 🔍 Root Cause: Confusion between file I/O and network I/O, brittle parsers, memory model gaps

---

## 📚 Module Overview

### Module 1: HTTP Transport (30 min)
**Learning Objective:** Distinguish file I/O from network I/O

**You Will Learn:**
- Why `open('http://...')` fails (protocol boundary)
- How to use `requests.get()` correctly
- HTTP status code handling (404, 500)
- Error handling best practices

**Harden Bug:** Your correct code will be modified to use `open()` instead of `requests.get()`. You'll debug the `FileNotFoundError` and understand *why* it fails.

---

### Module 2: Data Parsing & Extraction (45 min)
**Learning Objective:** Build robust parsers using BeautifulSoup + Regex

**You Will Learn:**
- Why `split(',')` is brittle for real-world data
- How to use BeautifulSoup for DOM traversal
- Regex patterns that handle whitespace variations
- When to use parsing libraries vs string manipulation

**Harden Bug:** Your correct regex-based parser will be replaced with `split()`. You'll see it break on inconsistent whitespace and learn defensive parsing.

---

### Module 3: Grid Visualization (45 min)
**Learning Objective:** Master 2D list initialization and Python's memory model

**You Will Learn:**
- The reference copying trap: `[[' '] * w] * h`
- Why list comprehensions create independent rows
- 0-indexed coordinate calculations (`max_x + 1`)
- Row-major order (`grid[y][x]`)

**Harden Bug:** Your correct initialization will be replaced with the aliased version. When you place a character, it will appear in *all rows*. You'll debug shallow vs deep copying.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Basic programming knowledge
- Familiarity with functions and loops

### Installation
```bash
# From the repository root
uv run mastery init job_prep_data_annotation
```

### Workflow
```bash
# 1. View current challenge
uv run mastery show

# 2. Implement in cs336_basics/utils.py
# (Follow the build prompt instructions)

# 3. Submit for validation
uv run mastery submit

# 4. Answer justify questions
uv run mastery submit
# (Opens your $EDITOR for answers)

# 5. Debug the harden challenge
uv run mastery start-challenge
uv run mastery submit
```

---

## 📊 Expected Timeline

| Stage | Module 1 | Module 2 | Module 3 | Total |
|-------|----------|----------|----------|-------|
| **Build** | 15 min | 20 min | 20 min | 55 min |
| **Justify** | 10 min | 15 min | 15 min | 40 min |
| **Harden** | 10 min | 15 min | 15 min | 40 min |
| **Subtotal** | 35 min | 50 min | 50 min | **2h 15m** |

*Plus ~30 min reading and setup = **~3 hours total***

---

## 🎓 Learning Outcomes

After completing this curriculum, you will be able to:

### Technical Skills
✅ Fetch documents from URLs using `requests`  
✅ Parse HTML with BeautifulSoup  
✅ Extract data with regex patterns  
✅ Initialize 2D grids without reference bugs  
✅ Handle sparse-to-dense coordinate mapping  

### Conceptual Understanding
✅ Distinguish protocol layers (file system vs network)  
✅ Understand when string manipulation is insufficient  
✅ Explain Python's memory model (shallow vs deep copy)  
✅ Write defensive code that handles real-world variations  

### Assessment Readiness
✅ Complete the "Secret Message Decoder" problem type  
✅ Debug common Python pitfalls quickly  
✅ Write production-quality code under time pressure  

---

## 🔬 Pedagogical Approach

This curriculum uses the **Build-Justify-Harden (BJH)** loop:

1. **BUILD:** Implement a working solution from a spec
2. **JUSTIFY:** Answer Socratic questions to prove conceptual understanding
3. **HARDEN:** Debug a bug injected into *your* correct code

### Why This Works

**Traditional Tutorial:**
- Read about `requests` library
- Copy-paste an example
- Move on (no retention)

**Mastery Engine:**
- Implement `fetch_document()` yourself
- Explain why `open()` can't fetch URLs
- Debug code that tries to use `open()` and watch it fail
- **Result:** You'll never confuse file I/O and network I/O again

---

## 📖 Source Material

This curriculum was derived from:
- **30 Days of Python** (Days 11, 13, 18, 22, 28)
- **DataAnnotation Assessment Failure Analysis**
- **Real-World Web Scraping Patterns**

However, the content is **original** - we extracted concepts but wrote our own:
- Build prompts (Mastery Engine format)
- Validators (executable test suites)
- Justify questions (depth enforcement)
- Bugs (surgical failure injection)

---

## 🛠️ Technical Details

### Dependencies
- `requests` (HTTP library)
- `beautifulsoup4` (HTML parsing)
- `lxml` (BS4 parser backend)

### File Locations
Your implementations go in:
- `cs336_basics/utils.py`

Functions you'll implement:
- `fetch_document(url: str) -> str`
- `extract_coordinates(html: str) -> list[tuple[int, int, str]]`
- `render_grid(coords: list[tuple[int, int, str]]) -> list[str]`

---

## 🎯 Success Metrics

You've successfully completed this curriculum when:

1. ✅ All validators pass (green checkmarks)
2. ✅ All justify questions answered correctly
3. ✅ All harden bugs fixed
4. ✅ You can explain *why* each bug occurred
5. ✅ You feel confident tackling web scraping assessments

---

## 🔗 Related Curricula

**If you also need:**
- **Competitive Programming:** Try `cp_accelerator` (959 LeetCode-style problems)
- **Deep Learning:** Try `cs336_a1` (Transformer implementation)
- **General Python:** Try `python_for_cp` (Standard library mastery)

---

## 📞 Support

**Issues with the curriculum?**
- Check `~/.mastery_engine.log` for detailed errors
- Ensure `uv pip install -e .` completed successfully
- Verify `requests` and `beautifulsoup4` are installed

**Getting stuck on a module?**
- Re-read the build prompt carefully
- Look at the validator test names for hints
- Remember: The bug in the harden stage is intentional

---

## 📜 License & Attribution

**Curriculum Design:** Mastery Engine Team  
**Source Inspiration:** Asabeneh/30-Days-Of-Python (MIT License)  
**Engine:** Mastery Engine (Custom pedagogical platform)

---

**Ready to begin?**

```bash
uv run mastery init job_prep_data_annotation
uv run mastery show
```

Good luck! 🚀
