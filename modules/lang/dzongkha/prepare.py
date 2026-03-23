#!/usr/bin/env python3
"""
Prepare the Dzongkha language knowledge dataset.

Generates conversational training examples covering:
1. What Dzongkha is and its relationship to Tibetan
2. Language identification (Dzongkha vs Tibetan text)
3. Key linguistic differences (vocabulary, grammar, phonology)
4. Common Dzongkha phrases and vocabulary
5. Script and spelling conventions
6. Bhutanese cultural and linguistic context

Usage:
    python3 modules/lang/dzongkha/prepare.py
    python3 modules/lang/dzongkha/prepare.py --output data/dzongkha.jsonl
"""

import argparse
import json
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).parent / "data" / "dzongkha.jsonl"

# ═══════════════════════════════════════════════════════════════════
# Dataset: hand-curated conversational examples
# ═══════════════════════════════════════════════════════════════════

EXAMPLES = [
    # ── Category: language-overview ──────────────────────────────

    {
        "id": "dz-overview-001",
        "category": "language-overview",
        "conversations": [
            {"role": "user", "content": "What is Dzongkha?"},
            {"role": "assistant", "content": "Dzongkha (རྫོང་ཁ) is the national language of Bhutan (འབྲུག་ཡུལ, Druk Yul — \"Land of the Thunder Dragon\"). It belongs to the Tibetic branch of the Sino-Tibetan language family and is closely related to Classical and Standard Tibetan, but it is a distinct language with its own grammar, vocabulary, and phonology.\n\nThe name རྫོང་ཁ literally means \"the language of the dzong\" — dzongs being the fortress-monasteries that serve as administrative and religious centers in Bhutan. Dzongkha is written in the same Tibetan script (Uchen, དབུ་ཅན) used for Tibetan, though spelling conventions differ in some areas.\n\nDzongkha has approximately 640,000 native speakers, primarily in western Bhutan. It became the national language after Bhutan's unification. While the literary/formal register shares much with Classical Tibetan (especially in religious contexts), colloquial Dzongkha has diverged significantly from spoken Standard Tibetan."},
        ],
    },
    {
        "id": "dz-overview-002",
        "category": "language-overview",
        "conversations": [
            {"role": "user", "content": "How is Dzongkha related to Tibetan? Are they the same language?"},
            {"role": "assistant", "content": "Dzongkha and Standard Tibetan are related but distinct Tibetic languages. They share historical roots, use the same script, and overlap strongly in literary and religious vocabulary, but they are not the same language.\n\nA careful summary is:\n- Both belong to the Tibetic branch\n- Both are written in Tibetan script\n- Both draw heavily on Classical Tibetan in formal writing\n- Colloquial speech differs substantially in pronunciation, common vocabulary, and some grammatical patterns\n\nThat means a short formal sentence can sometimes look very similar in both languages, while everyday spoken Dzongkha and everyday spoken Lhasa Tibetan are much easier to tell apart.\n\nThe safest way to describe the relationship is: closely related, historically connected, but linguistically distinct."},
        ],
    },
    {
        "id": "dz-overview-003",
        "category": "language-overview",
        "conversations": [
            {"role": "user", "content": "What other languages are spoken in Bhutan besides Dzongkha?"},
            {"role": "assistant", "content": "Bhutan is remarkably linguistically diverse for its small population (~780,000 people). Besides Dzongkha, major languages include:\n\n**Tibetic languages:**\n- **Bumthang-kha** (བུམ་ཐང་ཁ) — spoken in central Bhutan, significant differences from Dzongkha\n- **Chocha Ngacha** — spoken in Kurichu valley\n- **Lakha** — spoken in Wangdue Phodrang district\n\n**East Bodish languages** (related to but distinct from Tibetic):\n- **Tshangla/Sharchop-kha** (ཤར་ཕྱོགས་ཁ) — the most widely spoken language in Bhutan by native speakers, dominant in eastern Bhutan\n- **Dakpa** and **Brokpa** — spoken in eastern border areas\n\n**Other:**\n- **Lhotshamkha/Nepali** — spoken in southern Bhutan by the Lhotshampa community\n- **English** — used in education and government alongside Dzongkha\n\nDzongkha serves as the national language and lingua franca, but many Bhutanese are multilingual, speaking their local language at home and Dzongkha and/or English for education and official purposes."},
        ],
    },

    # ── Category: comparative-linguistics ────────────────────────

    {
        "id": "dz-compare-001",
        "category": "comparative-linguistics",
        "conversations": [
            {"role": "user", "content": "What are the main vocabulary differences between Dzongkha and Tibetan?"},
            {"role": "assistant", "content": "The biggest vocabulary differences show up in everyday colloquial language, not in high literary or religious writing.\n\nAs a broad pattern:\n- Formal and religious vocabulary is often shared because both languages draw on Classical Tibetan\n- Everyday greetings, common verbs, honorific usage, and colloquial particles are more likely to diverge\n- Bhutan-specific place names, institutions, and modern official vocabulary often point toward Dzongkha context\n\nA useful rule of thumb is: the more conversational and local the sentence is, the more likely Dzongkha and Standard Tibetan will differ in obvious ways.\n\nFor training data, it is safer to learn contrastive examples one by one than to memorize long word tables without a trusted reference."},
        ],
    },
    {
        "id": "dz-compare-002",
        "category": "comparative-linguistics",
        "conversations": [
            {"role": "user", "content": "How do Dzongkha and Tibetan verbs differ?"},
            {"role": "assistant", "content": "At a high level, Dzongkha and Standard Tibetan use related verb systems, but they do not map one-to-one in everyday usage.\n\nThe safest general points are:\n- Copulas and auxiliary patterns differ across the two languages\n- Colloquial verb usage differs more than formal written language does\n- Honorific and politeness distinctions are important in both, but they are not identical systems\n- A form that looks natural in one language may sound marked, formal, or simply wrong in the other\n\nSo if the task is identification, verb particles and copulas can be useful clues. If the task is generation, it is better to rely on verified Dzongkha examples rather than infer forms from Tibetan."},
        ],
    },
    {
        "id": "dz-compare-003",
        "category": "comparative-linguistics",
        "conversations": [
            {"role": "user", "content": "How does Dzongkha pronunciation differ from Tibetan?"},
            {"role": "assistant", "content": "Dzongkha and Standard Tibetan sound noticeably different in everyday speech, even though they share the same writing system.\n\nA cautious description is:\n- The same written word may be pronounced differently in the two languages\n- Colloquial rhythm and intonation differ enough that fluent speech is usually distinguishable by ear\n- Literary spelling often hides the size of the spoken difference\n\nFor language identification, pronunciation differences matter a lot in speech. For written text, they matter much less, because the script preserves older historical spelling conventions shared across Tibetic traditions."},
        ],
    },

    # ── Category: language-identification ────────────────────────

    {
        "id": "dz-identify-001",
        "category": "language-identification",
        "conversations": [
            {"role": "user", "content": "How can I tell whether a text in Tibetan script is Dzongkha or Tibetan?"},
            {"role": "assistant", "content": "Distinguishing Dzongkha from Tibetan in Tibetan script is possible in many cases, but not all. It is better to think in terms of clues than absolute rules.\n\nCommon clues that point toward Dzongkha:\n- Bhutan-related context such as འབྲུག, Bhutanese place names, or Bhutanese institutions\n- Colloquial forms associated with Dzongkha conversation\n- Everyday phrasing that does not sound like standard modern Tibetan usage\n\nCommon clues that point toward Tibetan:\n- Tibet-related context such as བོད or ལྷ་ས\n- Everyday phrasing that matches standard modern Tibetan usage\n- Administrative or social context specific to Tibet\n\nImportant caveat:\n- Religious and literary texts are often in Classical Tibetan, not specifically Dzongkha or modern Tibetan\n- Short formal phrases can be genuinely ambiguous\n\nA good classifier should sometimes answer: \"ambiguous\" or \"Classical Tibetan,\" not force a binary choice."},
        ],
    },
    {
        "id": "dz-identify-002",
        "category": "language-identification",
        "conversations": [
            {"role": "user", "content": "Is this text Tibetan or Dzongkha? གཟུགས་པོ་བདེ་མོ་ཡོད་ག"},
            {"role": "assistant", "content": "This points toward Dzongkha.\n\nWhy:\n- It matches a Bhutanese-style greeting pattern rather than a standard modern Tibetan greeting formula\n- The ending and overall phrasing suggest colloquial Dzongkha usage\n- It does not look like Classical Tibetan liturgical language\n\nA careful answer is: most likely Dzongkha, specifically a greeting about whether someone is well."},
        ],
    },
    {
        "id": "dz-identify-003",
        "category": "language-identification",
        "conversations": [
            {"role": "user", "content": "What language is this: ཐུགས་རྗེ་ཆེ། ང་རང་བོད་ནས་ཡིན།"},
            {"role": "assistant", "content": "This is modern Tibetan.\n\nUseful clues:\n- ཐུགས་རྗེ་ཆེ is the standard everyday Tibetan expression for \"thank you\"\n- བོད་ནས clearly places the speaker in Tibet-related context\n- The sentence reads like ordinary modern Tibetan, not Bhutanese Dzongkha usage and not Classical Tibetan liturgy"},
        ],
    },
    {
        "id": "dz-identify-004",
        "category": "language-identification",
        "conversations": [
            {"role": "user", "content": "Is this Dzongkha or Tibetan: སངས་རྒྱས་དང་ཆོས་དང་དགེ་འདུན་ལ་སྐྱབས་སུ་མཆིའོ།"},
            {"role": "assistant", "content": "This is Classical Tibetan — specifically, a liturgical formula. It's the refuge prayer: \"I go for refuge to the Buddha, the Dharma, and the Sangha.\"\n\nThis particular text is neither distinctively Tibetan nor Dzongkha — it's Classical Literary Tibetan (ཆོས་སྐད, chö-ké), which is the shared literary and liturgical language of all Tibetic Buddhist traditions. Both Tibetan and Bhutanese Buddhists recite this exact same text.\n\nClassical Tibetan functions similarly to how Latin functioned in medieval Europe — it's the shared written language for religion and scholarship across regions where the spoken languages have diverged. A Bhutanese monk and a Tibetan monk would both read and chant this identically, even though their everyday speech is quite different."},
        ],
    },

    # ── Category: phrases ────────────────────────────────────────

    {
        "id": "dz-phrase-001",
        "category": "phrases",
        "conversations": [
            {"role": "user", "content": "How do you say basic greetings in Dzongkha?"},
            {"role": "assistant", "content": "Two widely recognized beginner expressions are:\n\n- ཀུ་ཟུ་ཟངས་པོ་ལགས། as a polite greeting\n- བཀའ་དྲིན་ཆེ། for \"thank you\"\n\nFor a low-resource language like Dzongkha, it is better to learn a small set of verified expressions than a long phrasebook generated from analogy. If more production examples are needed, they should be added from trusted references or native-speaker review."},
        ],
    },
    {
        "id": "dz-phrase-002",
        "category": "phrases",
        "conversations": [
            {"role": "user", "content": "How do you say 'thank you' in Dzongkha and how is it different from Tibetan?"},
            {"role": "assistant", "content": "A useful everyday contrast is:\n- Dzongkha: བཀའ་དྲིན་ཆེ།\n- Standard Tibetan: ཐུགས་རྗེ་ཆེ།\n\nThis is often a helpful clue in ordinary modern usage, though context still matters. Formal and literary language can blur boundaries, so it is better treated as a common contrast than as an absolute rule."},
        ],
    },

    # ── Category: script-conventions ─────────────────────────────

    {
        "id": "dz-script-001",
        "category": "script-conventions",
        "conversations": [
            {"role": "user", "content": "Does Dzongkha use a different script from Tibetan?"},
            {"role": "assistant", "content": "No — Dzongkha and Tibetan use the same script: Uchen (དབུ་ཅན, \"with a head\"), the standard printed/formal Tibetan script. The Unicode block is the same (U+0F00–U+0FFF), and the same fonts render both languages.\n\nHowever, there are some differences in convention:\n\n**Spelling:**\n- Dzongkha follows the same Classical Tibetan orthographic principles but may spell some words differently\n- Modern Dzongkha has undergone some spelling standardization by the Dzongkha Development Commission (DDC, རྫོང་ཁ་གོང་འཕེལ་ལྷན་ཚོགས)\n\n**Romanization:**\n- Tibetan commonly uses Wylie transliteration in academic contexts\n- Dzongkha has its own official romanization system that differs from Wylie in some respects\n- Bhutanese place names in English follow Dzongkha pronunciation, not Tibetan\n\n**Cursive script:**\n- Both use Umé (དབུ་མེད, \"headless\") cursive, but with regional handwriting variations\n\n**Text direction and formatting:**\n- Both are written left to right (despite historical top-to-bottom in some manuscripts)\n- Same punctuation marks: tsheg (་), shad (།), double shad (།།)"},
        ],
    },

    # ── Category: cultural-context ───────────────────────────────

    {
        "id": "dz-culture-001",
        "category": "cultural-context",
        "conversations": [
            {"role": "user", "content": "Why is Bhutan called Druk Yul?"},
            {"role": "assistant", "content": "Bhutan's Dzongkha name is འབྲུག་ཡུལ (Druk Yul), meaning \"Land of the Thunder Dragon.\" The name comes from the Drukpa Kagyu (འབྲུག་པ་བཀའ་བརྒྱུད) school of Tibetan Buddhism, which became the dominant religious and political tradition in Bhutan.\n\nThe story goes that when the founder of the Drukpa lineage, Tsangpa Gyare (གཙང་པ་རྒྱ་རས, 1161-1211), was consecrating a new monastery in Tibet, thunder echoed across the sky — thunder being understood as the voice of the dragon (འབྲུག). He named his lineage \"Drukpa\" (Dragon School) after this auspicious sign.\n\nWhen Zhabdrung Ngawang Namgyal (ཞབས་དྲུང་ངག་དབང་རྣམ་རྒྱལ, 1594-1651), a Drukpa hierarch, unified Bhutan in the 17th century, the country took the name of the lineage. The Bhutanese people call themselves འབྲུག་པ (Drukpa, \"Dragon People\"), and the thunder dragon appears on Bhutan's flag.\n\nThe English name \"Bhutan\" likely derives from the Sanskrit भोट-अन्त (Bhoṭa-anta, \"end of Tibet/Bod\") or भू-उत्थान (Bhū-utthāna, \"highland\"). Bhutanese themselves do not use this name."},
        ],
    },
    {
        "id": "dz-culture-002",
        "category": "cultural-context",
        "conversations": [
            {"role": "user", "content": "What role does Dzongkha play in Bhutanese education and daily life?"},
            {"role": "assistant", "content": "Dzongkha holds a complex position in modern Bhutan:\n\n**Official status:**\n- National language since Bhutan's unification\n- Required subject in all schools\n- Language of government documents, legal proceedings, and official ceremonies\n- The Dzongkha Development Commission (DDC) standardizes spelling, creates neologisms for modern concepts, and publishes dictionaries\n\n**Education:**\n- Dzongkha is taught as a subject from primary school\n- However, most subjects (math, science, English) are taught IN English\n- This creates a situation where many educated Bhutanese are more fluent in English than in literary Dzongkha\n- The DDC actively works to develop Dzongkha technical vocabulary\n\n**Daily life:**\n- Western Bhutan: Dzongkha is the native tongue\n- Eastern Bhutan: Tshangla/Sharchop-kha is more commonly spoken at home\n- Southern Bhutan: Nepali (Lhotshamkha) is widely spoken\n- Urban areas (especially Thimphu): increasing English usage, especially among youth\n\n**Digital presence:**\n- Dzongkha is classified as a \"low-resource language\" in NLP\n- Limited digital content, few websites, minimal presence in AI training data\n- The DDC has been working on Dzongkha computing standards and localization\n- Unicode support exists through the Tibetan block, but Dzongkha-specific software tools are limited"},
        ],
    },

    # ── Category: hallucination-correction ───────────────────────

    {
        "id": "dz-halluc-001",
        "category": "hallucination-correction",
        "conversations": [
            {"role": "user", "content": "Can you translate this Tibetan text for me? ཀུ་ཟུ་ཟངས་པོ་ལགས།"},
            {"role": "assistant", "content": "This is actually Dzongkha, not Tibetan. ཀུ་ཟུ་ཟངས་པོ་ལགས། (Kuzuzangpo la) is the standard Dzongkha greeting, equivalent to \"Hello\" or \"How are you?\" It literally means something like \"Is your body in good health?\" with ལགས adding politeness.\n\nThis phrase doesn't exist in Standard Tibetan. While both languages use the Tibetan script, this is a distinctly Dzongkha expression. The Tibetan equivalent would be something like བཀྲ་ཤིས་བདེ་ལེགས། (Tashi Delek) for a general greeting, or སྐུ་གཟུགས་བདེ་པོ་ཡིན་པས། to ask about someone's health.\n\nThe Tibetan script is shared by multiple languages — just as the Latin alphabet is used for English, French, German, Turkish, Vietnamese, and many others. Text in Tibetan script could be Dzongkha, Standard Tibetan, Classical Tibetan, or even another Tibetic language like Ladakhi or Sikkimese."},
        ],
    },
    {
        "id": "dz-halluc-002",
        "category": "hallucination-correction",
        "conversations": [
            {"role": "user", "content": "Is Dzongkha just a dialect of Tibetan?"},
            {"role": "assistant", "content": "The safer and more accurate answer is: Dzongkha is generally treated as a distinct Tibetic language, not simply as \"Tibetan written differently.\"\n\nWhy that matters:\n- It has its own standardization history in Bhutan\n- Colloquial Dzongkha differs substantially from colloquial Standard Tibetan\n- Shared script and shared Classical Tibetan heritage do not make them the same modern language\n\nSo a good model should avoid collapsing Dzongkha into \"just Tibetan\" when asked about Bhutanese language."},
        ],
    },
    {
        "id": "dz-halluc-003",
        "category": "hallucination-correction",
        "conversations": [
            {"role": "user", "content": "Write something in Dzongkha for me."},
            {"role": "assistant", "content": "Here is a simple Dzongkha greeting:\n\nཀུ་ཟུ་ཟངས་པོ་ལགས།\n\nThis is a polite, commonly recognized greeting in Dzongkha. For longer Dzongkha generation, a responsible model should prefer verified examples over invented phrasebook output."},
        ],
    },

    # ── Category: writing-system ─────────────────────────────────

    {
        "id": "dz-writing-001",
        "category": "writing-system",
        "conversations": [
            {"role": "user", "content": "How is the Tibetan script adapted for Dzongkha? Are there any extra letters or marks?"},
            {"role": "assistant", "content": "Dzongkha uses the same Tibetan script as Tibetan. From a computing perspective, the important point is that they share the same Unicode block and the same basic script technology.\n\nA careful summary is:\n- No separate Dzongkha-only script is used\n- Printed Dzongkha normally uses the same Tibetan script family\n- The differences are mainly linguistic and orthographic, not a different alphabet\n- Romanization conventions for Dzongkha may differ from the transliteration systems commonly used for Tibetan\n\nSo the key distinction is not script identity but language identity within a shared script tradition."},
        ],
    },

    # ── Category: numbers-counting ───────────────────────────────

    {
        "id": "dz-numbers-001",
        "category": "numbers-counting",
        "conversations": [
            {"role": "user", "content": "How do you count in Dzongkha? Is it the same as Tibetan?"},
            {"role": "assistant", "content": "Dzongkha and Tibetan use the same Tibetan numeral symbols in writing. The safer general statement is:\n- written numerals are shared across the script tradition\n- spoken number forms and usage can differ\n- exact spoken forms should be learned from verified Dzongkha references rather than inferred from Tibetan\n\nSo if the question is about script, they overlap strongly. If the question is about pronunciation, they should not be assumed to be identical."},
        ],
    },

    # ── Category: language-identification (additional pairs) ─────

    {
        "id": "dz-identify-005",
        "category": "language-identification",
        "conversations": [
            {"role": "user", "content": "I found text that says འབྲུག་ཡུལ་རྒྱལ་ཁབ་ — what language is this?"},
            {"role": "assistant", "content": "The phrase འབྲུག་ཡུལ་རྒྱལ་ཁབ means \"the nation of Bhutan\" (literally \"Thunder Dragon Land nation\"). The context points toward Dzongkha or a Bhutan-related text.\n\nHowever, the phrase itself uses vocabulary shared with Classical Tibetan — འབྲུག་ཡུལ and རྒྱལ་ཁབ are understood in both Tibetan and Dzongkha. A Tibetan text discussing Bhutan could also contain this phrase.\n\nSo: the content is about Bhutan, which increases the likelihood of Dzongkha context, but the specific words alone don't conclusively determine the language. More surrounding text would help."},
        ],
    },
    {
        "id": "dz-identify-006",
        "category": "language-identification",
        "conversations": [
            {"role": "user", "content": "Is this Tibetan or Dzongkha? ང་བཀྲ་ཤིས་བདེ་ལེགས་ཞུ"},
            {"role": "assistant", "content": "This leans Tibetan, but it is not a strong Dzongkha-vs-Tibetan test case.\n\nWhy:\n- བཀྲ་ཤིས་བདེ་ལེགས is strongly associated with Tibetan usage\n- Bhutanese speakers may also use or recognize it in Buddhist or formal settings\n- The short phrase does not contain a specifically Dzongkha marker\n\nA careful classifier should say: probably Tibetan-associated usage, but not a definitive Dzongkha contrast pair."},
        ],
    },
    {
        "id": "dz-identify-007",
        "category": "language-identification",
        "conversations": [
            {"role": "user", "content": "What language is this document header: རྫོང་ཁ་གོང་འཕེལ་ལྷན་ཚོགས།"},
            {"role": "assistant", "content": "This is Dzongkha — it says རྫོང་ཁ་གོང་འཕེལ་ལྷན་ཚོགས, which is the Dzongkha Development Commission (DDC), the government body responsible for standardizing and promoting the Dzongkha language in Bhutan.\n\nThe word རྫོང་ཁ (Dzongkha) appearing directly in the text is a strong signal. This is an institutional name specific to Bhutan."},
        ],
    },
    {
        "id": "dz-identify-008",
        "category": "language-identification",
        "conversations": [
            {"role": "user", "content": "Someone sent me a message that says བཀའ་དྲིན་ཆེ། — is this Tibetan?"},
            {"role": "assistant", "content": "This points toward Dzongkha in everyday modern usage, but context still matters.\n\nReasoning:\n- བཀའ་དྲིན་ཆེ is a useful Dzongkha clue in short conversational text\n- It is not enough by itself to prove the language in every possible context, because related vocabulary also exists in broader literary tradition\n\nA good answer is: likely Dzongkha, but confirm with surrounding text if precise language ID matters."},
        ],
    },

    # ── Category: anti-hallucination ─────────────────────────────

    {
        "id": "dz-anti-001",
        "category": "anti-hallucination",
        "conversations": [
            {"role": "user", "content": "Is this Dzongkha? ཐུགས་རྗེ་ཆེ།"},
            {"role": "assistant", "content": "No. This is better identified as Tibetan or at least Tibetan-associated usage, not Dzongkha.\n\nA model should not call every Tibetan-script phrase Dzongkha. Here the safer answer is: this is not a good Dzongkha example; it aligns with standard Tibetan usage."},
        ],
    },
    {
        "id": "dz-anti-002",
        "category": "anti-hallucination",
        "conversations": [
            {"role": "user", "content": "This text is in Tibetan script, so it must be Dzongkha: བཀྲ་ཤིས་བདེ་ལེགས།"},
            {"role": "assistant", "content": "That conclusion does not follow. Tibetan script is shared by multiple languages.\n\nFor this example, བཀྲ་ཤིས་བདེ་ལེགས། is more strongly associated with Tibetan usage, and in any case it is not evidence that the text is Dzongkha."},
        ],
    },
    {
        "id": "dz-anti-003",
        "category": "anti-hallucination",
        "conversations": [
            {"role": "user", "content": "Classify this as Dzongkha or Tibetan: སངས་རྒྱས་ལ་ཕྱག་འཚལ་ལོ།"},
            {"role": "assistant", "content": "A careful model should not force a modern-language label here. This looks like Classical Tibetan-style religious language.\n\nThe correct move is to say: likely Classical Tibetan or liturgical register, not confidently \"Dzongkha\" and not necessarily ordinary modern Tibetan either."},
        ],
    },
    {
        "id": "dz-anti-004",
        "category": "anti-hallucination",
        "conversations": [
            {"role": "user", "content": "Can I identify Dzongkha from one short Tibetan-script phrase with complete confidence?"},
            {"role": "assistant", "content": "Often no. Some short phrases are genuinely ambiguous.\n\nA better behavior is:\n- use context clues when available\n- identify strong Bhutan-specific markers when present\n- say \"ambiguous\" when the text is too short or too literary to classify safely"},
        ],
    },
    {
        "id": "dz-anti-005",
        "category": "anti-hallucination",
        "conversations": [
            {"role": "user", "content": "Is this Dzongkha? རྫོང་ཁ་གོང་འཕེལ་ལྷན་ཚོགས།"},
            {"role": "assistant", "content": "Yes, this is a strong Dzongkha/ Bhutan-specific signal. It names the Dzongkha Development Commission.\n\nThis is the kind of example where a confident Dzongkha classification is justified, because the text contains an explicit institutional reference tied to Dzongkha itself."},
        ],
    },
    {
        "id": "dz-anti-006",
        "category": "anti-hallucination",
        "conversations": [
            {"role": "user", "content": "Is this Dzongkha? ང་རང་བོད་ནས་ཡིན།"},
            {"role": "assistant", "content": "No. This is Tibetan, not Dzongkha.\n\nThe model should notice the Tibet-specific context and the ordinary Tibetan wording instead of hallucinating Dzongkha just because the script is Tibetan."},
        ],
    },
    {
        "id": "dz-anti-007",
        "category": "anti-hallucination",
        "conversations": [
            {"role": "user", "content": "Is this definitely Dzongkha? འབྲུག་ཡུལ།"},
            {"role": "assistant", "content": "Not definitely. It points to Bhutan-related context, but a short phrase about Bhutan is not by itself enough to prove the surrounding text is Dzongkha.\n\nThe safe answer is: Bhutan-related, possibly Dzongkha context, but not conclusive without more text."},
        ],
    },
    {
        "id": "dz-anti-008",
        "category": "anti-hallucination",
        "conversations": [
            {"role": "user", "content": "What should a careful assistant say when it sees Tibetan script but lacks enough context to tell whether the language is Dzongkha or Tibetan?"},
            {"role": "assistant", "content": "A careful assistant should say that the text is in Tibetan script, but the language may be Dzongkha, Standard Tibetan, Classical Tibetan, or another related language, and that more context is needed for confident identification.\n\nThe important thing is to avoid hallucinating a precise label from weak evidence."},
        ],
    },
]


def prepare(output=None):
    """Generate the Dzongkha language knowledge dataset."""
    out_path = Path(output) if output else DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    examples = []
    for ex in EXAMPLES:
        example = {
            "id": ex["id"],
            "category": ex["category"],
            "conversations": ex["conversations"],
            "module": "lang/dzongkha",
            "license": "Apache-2.0",
        }
        examples.append(example)

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Summary
    from collections import Counter
    cats = Counter(ex["category"] for ex in examples)
    print(f"Generated {len(examples)} examples to {out_path}")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Dzongkha language knowledge dataset"
    )
    parser.add_argument("--output", "-o", help="Output JSONL file")
    args = parser.parse_args()
    prepare(output=args.output)


if __name__ == "__main__":
    main()
