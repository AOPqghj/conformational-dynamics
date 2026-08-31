# Compute disclosure

Reported work used local CPU/MPS hardware and Google Colab accelerators, including L4 GPUs and v5e-1 TPUs for embedding generation. CPU confounder fits were generally limited to two threads. Exact wall-clock time, peak memory, and total project compute were not recorded consistently for every historical run; the paper checklist therefore does not claim complete per-experiment compute disclosure.

For replication, the dominant costs are residue-embedding generation and neural model/SAE fitting. Pooled linear and tree fits are CPU-friendly; CNN, MHA, and SAE runs benefit from GPU acceleration. Checkpointed notebooks are supplied for long BioEMU jobs so interrupted cloud sessions can resume without repeating completed proteins.
