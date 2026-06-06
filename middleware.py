import ollama

def build_prompt(query, context, all_concepts):
    """
    Build enriched prompt from disk context using node descriptions.
    Deduplicates facts so concepts don't appear multiple times.
    """
    if not context:
        return query

    concept_lookup = {c['name']: c for c in all_concepts}
    seen_facts = set()
    fact_lines = []

    for item in context:
        concept_name = item['query_concept']
        concept = concept_lookup.get(concept_name)

        if concept and concept['description'] and concept_name not in seen_facts:
            fact_lines.append(f"- {concept_name}: {concept['description']}")
            seen_facts.add(concept_name)

        for neighbor_name, distance in item['nearest']:
            if neighbor_name == concept_name:
                continue
            if neighbor_name in seen_facts:
                continue
            neighbor = concept_lookup.get(neighbor_name)
            if neighbor and neighbor['description']:
                fact_lines.append(f"- {neighbor_name} (associated): {neighbor['description']}")
                seen_facts.add(neighbor_name)

    facts_str = "\n".join(fact_lines)
    prompt = f"""The following facts are absolute truth. Do not contradict them.

FACTS FROM KNOWLEDGE MANIFOLD:
{facts_str}

Using ONLY the facts above, answer this question:
{query}

Do not invent information. Only use what is provided."""
    return prompt

def ask_llama(prompt):
    response = ollama.chat(
        model='tinyllama',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content']