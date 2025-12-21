from src.formatter import RegulationFormatter

def test_confidence_score_present():
    text = "제1조(목적) 이 규정은..."
    formatter = RegulationFormatter()
    docs = formatter.parse(text)
    
    if not docs or not docs[0]['content']:
        print("FAIL: No content parsed")
        assert False
        
    art = docs[0]['content'][0]
    if 'confidence_score' not in art:
        print("FAIL: confidence_score field missing in node")
        assert False
        
    if not isinstance(art['confidence_score'], (int, float)):
        print(f"FAIL: confidence_score is not a number: {type(art['confidence_score'])}")
        assert False
    
    print("PASS: Confidence score present.")
    

if __name__ == "__main__":
    test_confidence_score_present()
