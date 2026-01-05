# Two-Room Memory Architecture

*An Efficient Context Management System for LLM Memory*

**Zach & Claude | January 2026**

---

## Overview

This document describes an architecture for efficient LLM memory management based on a two-room model with unidirectional information flow. The system defaults to treating all user information as important, filters on triviality rather than significance, and organizes stored information by relational posture rather than data category.

---

## Core Philosophy

The fundamental design inversion: instead of asking "is what the user said important?" the system asks "is what the user said trivial?" This reflects a relational posture where the user IS important by default, and we only gate information that demonstrably does not contribute to a meaningful relationship.

This approach is more computationally efficient because triviality is a tighter semantic cluster than meaningfulness. Encyclopedic queries, small talk, and generic factoids share common characteristics. Meaningful human information is sprawling and hard to bound.

---

## Architecture

### Room 1 (RAM)

The active context buffer. All input enters here via a single entrance point (unidirectional flow). Room 1 functions as disposable, high-bandwidth working memory.

#### The Triviality Gate

Each exchange E is compared against A_t (the triviality archetype) using cosine similarity:

```
R(E) = cosine_similarity(embed(E), A_t)

if R(E) ≥ θ_trivial → flush after response
else → persist to Room 2
```

The triviality archetype is constructed from canonical examples of non-relational exchanges:

```
A_t = mean([
    embed("What color are ladybugs"),
    embed("How many feet in a mile"),
    embed("What's the weather"),
    embed("Define photosynthesis"),
    embed("Random fact about penguins")
])
```

#### Retroactive Linking

To handle cases where context arrives after a seemingly trivial exchange (e.g., "Why are ladybugs red?" followed by "My mother loved ladybugs and she died yesterday"), the system performs backward scanning when high-relevance content lands.

Rather than maintaining a separate soft-delete buffer (which adds overhead on every exchange), retroactive linking only triggers on persist events. When something important is being compressed for Room 2, the compression function scans recent Room 1 context and absorbs related exchanges into a unified memory entry.

This is more efficient: O(1) for trivial exchanges, O(n) only when persistence occurs.

---

### Room 2 (SSD)

Persistent storage organized as a matrix structure indexed by relational category and weight band.

#### Two-Tier Storage

**Tier 1 (Immutable):** Information that won't change. Compressed flat with no metadata overhead. Conflicts trigger clarification flags (something unusual is happening).

**Tier 2 (Mutable):** Information that will or could change. Stored with timestamp, volatility score, and optional trajectory log. Conflicts trigger updates (expected behavior).

```
memory_entry = {
    fact: string,
    tier: 1 | 2,
    volatility: float (0-1),  // Tier 2 only
    last_updated: timestamp | null,
    trajectory: [] | null  // depth scales with volatility
}
```

---

### Relational Categories

Rather than organizing by data type (identity, family, career), Room 2 is indexed by relational posture—what the information demands from the system when engaging with the user.

| Category | Relational Demand | Examples |
|----------|-------------------|----------|
| **EMPATHY** | Emotional attunement, care, remembrance | "My mom died yesterday" |
| **UNDERSTANDING** | Accommodation, patience, adaptation | "I have ADHD" |
| **RESPECT** | Recognition of capability, no condescension | "I went to law school" |
| **COMMUNICATION** | Behavioral adjustment in engagement style | "I prefer direct communication" |
| **CONTEXT** | Useful if relevant, not load-bearing | "I work as a contractor" |
| **VOLATILE** | Important now, likely to change, hold loosely | "I'm shipping my game in January" |

This indexing strategy makes retrieval intent-aligned. When parsing user input, the system asks "what does this moment require from me?" and retrieves directly by category and weight band. One lookup, not two.

---

## Tier Assignment

Tier assignment combines objective statistical base rates with subjective syntactic analysis of the user's corpus.

```
tier_assignment(fact) = {
    base_mutability = population_probability(fact_category changes)
    user_signal = syntactic_volatility(fact_category, user_corpus)
    adjusted_mutability = (α * base_mutability) + (β * user_signal)

    if adjusted_mutability < θ_immutable:
        tier = 1
    else:
        tier = 2
        volatility = adjusted_mutability
}
```

The α and β weights scale with corpus size, allowing the system to self-tune:

```
early relationship:  α = 0.8, β = 0.2  // lean on base rates
mature relationship: α = 0.3, β = 0.7  // lean on observed patterns
```

---

## Syntactic Analysis

The subjective component of tier assignment is derived from linguistic and emotional patterning within the user's corpus. This is essentially psychological profiling in service of understanding—reading how they talk, not just what they say.

### Syntactic Profile Structure

```
syntactic_profile(user_corpus) = {
    // Lexical
    word_freq: map(word → count),
    hedge_ratio: count(hedging_words) / total_words,
    absolutist_ratio: count(absolutist_words) / total_words,

    // Structural
    avg_sentence_length: float,
    question_ratio: count(questions) / total_statements,
    self_reference_ratio: count(I/me/my) / total_words,
    tense_distribution: {past: float, present: float, future: float},

    // Emotional
    sentiment_by_topic: map(topic → avg_sentiment),
    intensity_ratio: count(intensifiers) / total_words,

    // Relational
    people_reference_style: map(person → name | role | both),
    attachment_valence: map(person/topic → pos | neg | ambiv)
}
```

### Volatility Derivation

```
volatility(fact_category) = weighted_sum(
    w1 * hedge_ratio_around(fact_category),
    w2 * sentiment_variance_around(fact_category),
    w3 * change_language_frequency(fact_category),
    w4 * tense_skew(fact_category),
    w5 * mention_frequency_trend(fact_category)
)
```

If insufficient data exists for a fact category, the system falls back to population base rates until the corpus is rich enough for personalized inference.

---

## Retrieval Mechanism

The matrix structure (category × weight band) enables O(1) retrieval. When new input arrives:

1. Parse input for required response posture (what does this moment demand?)
2. Identify relevant categories (EMPATHY, UNDERSTANDING, VOLATILE, etc.)
3. Go directly to category rows, prioritize high-weight columns
4. Surface relevant context without full scan

**Example:** User says "I'm nervous about the launch." System parses emotional disclosure + project reference, retrieves from EMPATHY (support context), VOLATILE (Airline Ops details), UNDERSTANDING (ADHD/perpetual motion brain), and COMMUNICATION (direct style preference). Response posture: be direct, acknowledge nerves without coddling, reference the specific project.

---

## System Summary

```
ROOM 1 (RAM):
  - All input enters here (unidirectional)
  - Triviality gate: compare to A_t
  - If similar → flush after response
  - If not → persist to Room 2
  - Retroactive linking on high-relevance input

ROOM 2 (SSD):
  - Matrix: [relational_category][weight_band]
  - Categories: EMPATHY, UNDERSTANDING, RESPECT,
               COMMUNICATION, CONTEXT, VOLATILE
  - Two tiers: Immutable / Mutable
  - Tier assignment: objective + subjective,
                     weighted by corpus maturity
  - Compression: flat (T1) or with trajectory (T2)

RETRIEVAL:
  - Parse input for response posture
  - Direct lookup: category + weight band
  - O(1) access

SYNTACTIC PROFILE:
  - Passive build from corpus
  - Informs volatility and tier assignment
  - Influence scales with corpus size
```

---

## Future Considerations

1. **Decay/Pruning:** Even Room 2 isn't infinite. Long-term maintenance rules for low-access, low-weight entries.

2. **Explicit Overrides:** User says "remember this" (override gate) or "forget this" (override persist).

3. **Weight Tuning:** Empirical refinement of w1-w5 in volatility derivation and α/β in tier assignment.

4. **Cross-Session Learning:** Using retrieval access patterns to refine the triviality archetype over time.
