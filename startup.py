from pathlib import Path

from tokenizer import LearnedTokenizer


ROOT = Path(
    __file__
).resolve().parent


def ensure_runtime():

    tokenizer = LearnedTokenizer(
        ROOT / "verdant_tokenizer.json"
    )

    if tokenizer.vocab_size >= 512:
        return

    seed = (
        """
        Verdant is a learned artificial intelligence system.
        Language can express ideas, relationships, actions, questions,
        explanations, and observations.

        Learning requires prediction, feedback, revision,
        generalization, and repeated testing.

        Conversation requires understanding context, references,
        follow-up questions, corrections, and changes of topic.

        Programming involves algorithms, data structures,
        syntax, functions, state, testing, debugging, and iteration.

        Science uses observation, measurement, hypotheses,
        evidence, models, prediction, and revision.

        Mathematics uses quantities, relationships,
        structures, operations, proof, and problem solving.
        """
        * 100
    )

    tokenizer.build(
        seed,
        max_vocab=8192,
    )


if __name__ == "__main__":
    ensure_runtime()
    print(
        "Verdant runtime initialized."
    )
