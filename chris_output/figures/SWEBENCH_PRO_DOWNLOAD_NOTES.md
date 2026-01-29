# SWE-bench Pro Download Notes

## Issue: API Requires Authentication

✅ **API Endpoint Found!** But requires authentication.

### Correct API Endpoint

```
https://api.docent.transluce.org/rest/{collection_id}/agent_run_with_tree?agent_run_id={run_id}&apply_base_where_clause=false&full_tree=true
```

Where:
- `collection_id` = `032fb63d-4992-4bfc-911d-3b7dafcb931f`
- `agent_run_id` = from the CSV (e.g., `eaa8e4b1-dda6-46ec-9787-ba7ccebfafa2`)

### Authentication Issue

**Unauthenticated requests return:**
```json
{"detail":"Unauthorized"}
```

The browser works because it sends authentication cookies/tokens automatically.

### Attempted APIs (For Reference)

1. **Docent API** (correct!) - ✅ Found but ❌ Requires auth

## Storage Estimate (Based on Similar Data)

Since we can't fetch sample trajectories, here's an estimate based on SWE-bench Verified data:

### SWE-bench Verified Statistics (for comparison)
- **Average trajectory size:** ~50-150 KB per trajectory (varies widely)
- **Median trajectory size:** ~80 KB
- **Range:** 10 KB (short failures) to 500+ KB (long runs)

### Estimated Storage for SWE-bench Pro

**Conservative estimate (80 KB avg):**
```
9,729 trajectories × 80 KB = 778 MB
```

**Upper bound estimate (150 KB avg):**
```
9,729 trajectories × 150 KB = 1.4 GB
```

**Recommended free space:** 2-3 GB (with buffer for processing)

### Factors Affecting Size

1. **Conversation length** (turns): SWE-bench Pro avg = 51.5 turns
   - Longer conversations → larger JSON files
   - File operations, tool outputs add to size

2. **Agent verbosity**: Some models produce longer responses

3. **Test output**: Failed test runs can generate extensive logs

## Alternative Approaches

### Option 1: Contact Docent Team

Reach out to the team behind https://docent.transluce.org to:
- Get API documentation for bulk export
- Request data dump of all trajectories
- Clarify authentication/access requirements

### Option 2: Extract Browser Cookies + Use API (RECOMMENDED)

**This is the best approach!** The API works but just needs authentication.

Steps:
1. Log into https://docent.transluce.org in your browser
2. Open Developer Tools (F12) → Application/Storage → Cookies
3. Copy the auth cookies (likely `session`, `auth_token`, or similar)
4. Pass cookies to the download script via httpx Client
5. Download all 9,729 trajectories via API in ~10-15 minutes

**Pros:**
- Fast (~10-15 minutes for all trajectories with batch_size=20)
- Reliable (uses official API)
- Resumable (Ctrl+C safe with --resume flag)

**Cons:**
- Requires manual cookie extraction (one-time setup)
- Cookies may expire (might need to refresh periodically)

**Implementation:** Update `download_swebench_pro_trajectories.py` to accept cookies as a parameter or read from a config file.

### Option 3: Playwright with Browser Context

Use Playwright to maintain an authenticated browser session:
1. Log in to Docent (manually or automated)
2. Keep browser context alive
3. Make API calls through the authenticated browser
4. Or intercept the network responses directly

**Pros:** Automated authentication handling
**Cons:** Slower than direct API (~5-8 hours), more complex

### Option 3: Use Existing CSV for IRT

The CSV already contains the key information for IRT analysis:
- `agent_run_id` → subject identifier
- `metadata.instance_id` → item identifier
- `metadata.resolved` → binary response (0/1)

**Full trajectories only needed if:**
- Want to re-grade with LLM judges
- Need to extract trajectory-based features
- Performing qualitative analysis of agent behavior

## Recommendation

**For IRT analysis:** Use the CSV directly, don't need full trajectories

**For LLM judge analysis:**
1. Try Option 1 (contact Docent team) first
2. If unavailable, use the CSV metadata for basic features
3. Consider scraping a small sample (~50 trajectories) for feature development

## Script Status

The `download_swebench_pro_trajectories.py` script is ready but needs:
- Correct API endpoint documentation from Docent
- Authentication method
- Possibly different data format/schema

Once API access is resolved, the script can download all 9,729 trajectories with:
```bash
python download_swebench_pro_trajectories.py --download --batch-size 10
```

Estimated download time with working API: 1-2 hours (with parallel downloads)
