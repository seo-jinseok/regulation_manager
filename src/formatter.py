import re
import json
from typing import List, Dict, Any

class RegulationFormatter:
    """
    Parses regulation text into structured JSON compatible with Korean Law Information Center structure.
    Hierarchy:
    - 조 (Article): "제1조(목적)"
    - 항 (Paragraph): "①", "1." (sometimes)
    - 호 (Item): "1.", "가."
    """

    def parse(self, text: str) -> List[Dict[str, Any]]:
        lines = text.split('\n')
        
        regulations = []
        
        # State capturing
        # We need to capture lines into specific buckets based on where we are.
        # But for multi-regulation files, the boundary is fuzzy until we hit "Article 1".
        # Strategy:
        # 1. Collect everything into 'current_regulation' dict buckets.
        # 2. When 'Article 1' is found, we assume a new start. 
        #    BUT we must retroactively check if the 'End' of the previous buffer claims to be the 'Title' of the new one.
        
        current_data = {
            "preamble": [],
            "articles": [],
            "appendices": []
        }
        
        current_article = None
        current_paragraph = None
        
        # Mode: 'PREAMBLE' | 'ARTICLES' | 'APPENDICES'
        mode = 'PREAMBLE' 

        def flush_regulation(next_preamble_lines: List[str] = None):
            nonlocal current_article, current_paragraph, mode, current_data
            
            # Close last article
            if current_article:
                current_data["articles"].append(current_article)
                current_article = None
                current_paragraph = None
            
            # Construct Doc
            if current_data["articles"] or current_data["preamble"] or current_data["appendices"]:
                reg = {
                    "preamble": "\n".join(current_data["preamble"]).strip(),
                    "articles": current_data["articles"],
                    "appendices": "\n".join(current_data["appendices"]).strip()
                }
                regulations.append(reg)
            
            # Reset
            current_data = {
                "preamble": next_preamble_lines if next_preamble_lines else [],
                "articles": [],
                "appendices": []
            }
            mode = 'PREAMBLE' if next_preamble_lines else 'PREAMBLE' # Reset to preamble for new doc
            
            # If we shifted lines, it means we already have content for the new doc, 
            # effectively we are in PREAMBLE mode waiting for Article 1.

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 1. Detect Article 1 (Potential Split)
            # Regex for "제1조" or "제 1 조"
            article_1_match = re.match(r'^(제\s*1\s*조)\s*(?:\(([^)]+)\))?\s*(.*)', line)
            
            if article_1_match:
                # This is Article 1.
                # If we have existing content, this is likely a Split Point.
                if current_data["articles"] or current_article:
                     # Heuristic: Find Regulation Title in the *tail* of the previous section.
                     # The previous section is either 'appendices' (if we were in APPENDICES mode) 
                     # or 'preamble' (unlikely if we had articles) or just 'articles' (if no appendices).
                     
                     # Usually, 'appendices' comes before the next 'Article 1'.
                     # Search target: current_data["appendices"]
                     
                     split_idx = -1
                     target_list = current_data["appendices"] if mode == 'APPENDICES' else current_data["preamble"] 
                     # Note: If mode was ARTICLES, we haven't started appendices/new preamble essentially? 
                     # Actually, often there is a Title line between last article and next Article 1.
                     # If mode is ARTICLES, the Title line would have been captured as... raw text in the last article? 
                     # OR we need to verify where non-matching lines go.
                     
                     # Let's look at where "Title" lines end up.
                     # If mode == APPENDICES: Title lines go to appendices.
                     # If mode == ARTICLES: Title lines go to last article content? (This is bad)
                     
                     # Refinement: We need a buffer for "Potential New Header" if we are in ARTICLES mode?
                     pass

                     # SEARCH BACKWARDS for a Title Line
                     # Title Pattern: Ends with "규정", "세칙", "지침", "요령" etc.
                     # Expanded list based on typical university regulations
                     title_candidates = ["규정", "세칙", "지침", "요령", "강령", "내규", "학칙", "헌장", "기준", "수칙", "준칙", "요강", "운영"]
                     
                     # We only search in 'appendices' since that's where "inter-regulation" text falls 
                     # if it didn't match an article.
                     start_next_lines = []
                     
                     if mode == 'APPENDICES':
                         # Search backwards
                         for i in range(len(current_data["appendices"]) - 1, -1, -1):
                             txt = current_data["appendices"][i].strip()
                             # Check if looks like a title
                             # 1. Ends with candidate
                             # 2. Not too long (Relaxed to 100)
                             # 3. Not a file path/url (defensive)
                             # 4. Or if it contains '규정' inside and is very short?
                             
                             if (any(txt.endswith(c) for c in title_candidates) or "규정" in txt) and len(txt) < 100:
                                 # Found split point!
                                 # i is the Title.
                                 # Everything from i onwards is NEXT doc.
                                 split_idx = i
                                 break
                         
                         if split_idx != -1:
                             start_next_lines = current_data["appendices"][split_idx:]
                             current_data["appendices"] = current_data["appendices"][:split_idx]
                     
                     flush_regulation(next_preamble_lines=start_next_lines)
            
            # 2. Detect Article (Generic)
            article_match = re.match(r'^(제\s*(\d+)\s*조)\s*(?:\(([^)]+)\))?\s*(.*)', line)
            if article_match:
                mode = 'ARTICLES' # Enter Article Mode
                
                # Close previous article
                if current_article:
                    current_data["articles"].append(current_article)
                
                article_no = article_match.group(2)
                article_title = article_match.group(3) or ""
                content = article_match.group(4)
                
                current_article = {
                    "article_no": article_no,
                    "title": article_title,
                    "content": [], 
                    "paragraphs": []
                }
                current_paragraph = None
                
                if content:
                    # Inline content handling
                    para_match = re.match(r'^([①-⑮])\s*(.*)', content)
                    if para_match:
                        current_paragraph = {
                            "paragraph_no": para_match.group(1),
                            "content": para_match.group(2),
                            "items": []
                        }
                        current_article["paragraphs"].append(current_paragraph)
                    else:
                        current_article["content"].append(content)
                        
                continue

            # 3. Detect Appendices (부칙)
            if re.match(r'^부\s*칙', line):
                mode = 'APPENDICES'
                # Close Article
                if current_article:
                    current_data["articles"].append(current_article)
                    current_article = None
                    current_paragraph = None
                
                current_data["appendices"].append(line)
                continue
                
            # 4. Handle Content based on Mode
            if mode == 'ARTICLES':
                # We are inside an article. Check for Structure (Paragraph, Item) 
                
                # Paragraph (①)
                para_match = re.match(r'^([①-⑮])\s*(.*)', line)
                if para_match and current_article:
                    current_paragraph = {
                        "paragraph_no": para_match.group(1),
                        "content": para_match.group(2),
                        "items": []
                    }
                    current_article["paragraphs"].append(current_paragraph)
                    continue

                # Item (1. or 가.)
                item_match = re.match(r'^(\d+\.|[가-하]\.)\s*(.*)', line)
                if item_match and current_article:
                    new_item = {"item_no": item_match.group(1), "content": item_match.group(2)}
                    if current_paragraph:
                        current_paragraph["items"].append(new_item)
                    else:
                        # Fallback: Treat as generic content of article
                         current_article["content"].append(line)
                    continue
                
                # Plain Text in ARTICLE mode
                if current_paragraph:
                    current_paragraph["content"] += " " + line
                elif current_article:
                    current_article["content"].append(line)
                else:
                    # Should not happen if logic is correct, but safe fallback
                    current_data["preamble"].append(line)

            elif mode == 'APPENDICES':
                # In Appendices, EVERYTHING is appendix content including bullets
                current_data["appendices"].append(line)
                
            elif mode == 'PREAMBLE':
                current_data["preamble"].append(line)

        # Final Flush
        flush_regulation()
        return regulations
