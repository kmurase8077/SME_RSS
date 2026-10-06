#!/usr/bin/env python3
"""
Retry AiZynthFinder while maintaining the skeleton through random C/N substitution
Dependencies: rdkit, aizynthfinder
"""

import argparse, json, random, subprocess, tempfile, os, sys, configparser
from pathlib import Path
from multiprocessing import Pool, cpu_count, Manager

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdchem
from rdkit import RDLogger

# Disable all RDKit warnings
RDLogger.DisableLog('rdApp.*')    # Use RDLogger.EnableLog('rdApp.*') to show them

# ---- User settings (Read from external file) ----
CONFIG_FILE = "rss_config.ini"
if "--ini" in sys.argv:
    try:
        idx = sys.argv.index("--ini")
        CONFIG_FILE = sys.argv[idx + 1]
    except IndexError:
        pass

# Default settings (Execution)
EXEC_SMILES = None
EXEC_BATCH = None
EXEC_CONFIG = "config.yml"
EXEC_SAMPLES = 60
EXEC_MAX_SWAPS = 4
EXEC_JOBS = max(1, cpu_count() // 2)
EXEC_OUT = "mutants_results.jsonl"

# Default settings (Settings)
MUTATE_ALLOWED = ["C", "N", "O", "S", "F", "Cl", "Br"]
MUTATE_WEIGHTS = [0.6, 0.2, 0.15, 0.02, 0.01, 0.01, 0.01]
MIN_N_RATIO = 0.10
MAX_N_RATIO = 0.30
MAX_COUNT_S = 1
MAX_COUNT_O = 2
MAX_COUNT_F = 1
MAX_COUNT_Cl = 1
MAX_COUNT_Br = 1
PROTECTED_SMARTS = [
    "[CX3](=O)[OX2H1,OX1-]",
    "[#6][CX3](=O)[#6]",
    "[OX2H]",
    "[NX3;H2;!$(NC=O)]",
    "[NX3;H1;!$(NC=O)]",
    "[#6][CX3](=O)[OX2H0][#6]",
    "[OD2]([#6])[#6]"
]

if os.path.exists(CONFIG_FILE):
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE, encoding='utf-8')

    if config.has_section('Execution'):
        if config.has_option('Execution', 'smiles') and config.get('Execution', 'smiles').strip():
            EXEC_SMILES = config.get('Execution', 'smiles').strip()
        if config.has_option('Execution', 'batch') and config.get('Execution', 'batch').strip():
            EXEC_BATCH = config.get('Execution', 'batch').strip()
        if config.has_option('Execution', 'config') and config.get('Execution', 'config').strip():
            EXEC_CONFIG = config.get('Execution', 'config').strip()
        if config.has_option('Execution', 'samples') and config.get('Execution', 'samples').strip():
            EXEC_SAMPLES = config.getint('Execution', 'samples')
        if config.has_option('Execution', 'max_swaps') and config.get('Execution', 'max_swaps').strip():
            EXEC_MAX_SWAPS = config.getint('Execution', 'max_swaps')
        if config.has_option('Execution', 'jobs') and config.get('Execution', 'jobs').strip():
            EXEC_JOBS = config.getint('Execution', 'jobs')
        if config.has_option('Execution', 'out') and config.get('Execution', 'out').strip():
            EXEC_OUT = config.get('Execution', 'out').strip()

    if config.has_section('Settings'):
        if config.has_option('Settings', 'mutate_allowed'):
            MUTATE_ALLOWED = [x.strip() for x in config.get('Settings', 'mutate_allowed').split(',')]
        if config.has_option('Settings', 'mutate_weights'):
            MUTATE_WEIGHTS = [float(x.strip()) for x in config.get('Settings', 'mutate_weights').split(',')]
        if config.has_option('Settings', 'min_n_ratio'):
            MIN_N_RATIO = config.getfloat('Settings', 'min_n_ratio')
        if config.has_option('Settings', 'max_n_ratio'):
            MAX_N_RATIO = config.getfloat('Settings', 'max_n_ratio')

    if config.has_section('MaxCounts'):
        if config.has_option('MaxCounts', 'S'): MAX_COUNT_S = config.getint('MaxCounts', 'S')
        if config.has_option('MaxCounts', 'O'): MAX_COUNT_O = config.getint('MaxCounts', 'O')
        if config.has_option('MaxCounts', 'F'): MAX_COUNT_F = config.getint('MaxCounts', 'F')
        if config.has_option('MaxCounts', 'Cl'): MAX_COUNT_Cl = config.getint('MaxCounts', 'Cl')
        if config.has_option('MaxCounts', 'Br'): MAX_COUNT_Br = config.getint('MaxCounts', 'Br')

    if config.has_section('ProtectedSMARTS'):
        PROTECTED_SMARTS = [val for key, val in config.items('ProtectedSMARTS')]

# ---- utility functions ----
def canonical_smiles(smi):
    m = Chem.MolFromSmiles(smi)
    if not m:
        return None
    return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)

def generic_scaffold_smi(mol):
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None:
        return None
    # Make it "generic" to smooth out atom differences (to compare skeletons = connections)
    g = MurckoScaffold.MakeScaffoldGeneric(scaf)
    return Chem.MolToSmiles(g, canonical=True)

def pick_cn_ring_indices(mol):
    """Return indices of C/N atoms belonging to aromatic rings."""
    idxs = []
    ri = mol.GetRingInfo()
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in ("C", "N"):
            continue
        if atom.GetIsAromatic() and atom.IsInRing():
            idxs.append(atom.GetIdx())
    return idxs

def try_cn_flip(mol, indices_to_flip):
    """Swap C<->N at specified indices and return Mol if it passes Sanitize."""
    em = Chem.RWMol(mol)
    for idx in indices_to_flip:
        a = em.GetAtomWithIdx(idx)
        sym = a.GetSymbol()
        if sym == "C":
            a.SetAtomicNum(7)  # C->N
            # Aromaticity is preserved (if RDKit cannot interpret it as [n], it will fail Sanitize)
        elif sym == "N":
            a.SetAtomicNum(6)  # N->C
        else:
            return None
    try:
        m2 = em.GetMol()
        Chem.SanitizeMol(m2)  # Exception if failed
        # Recalculate aromaticity (just in case)
        Chem.SetAromaticity(m2)
        return m2
    except Exception:
        return None

def check_constraints(mol):
    atoms = mol.GetAtoms()
    
    # 1. Check CN ratio
    c_count = sum(1 for a in atoms if a.GetSymbol() == "C")
    n_count = sum(1 for a in atoms if a.GetSymbol() == "N")
    denom = c_count + n_count
    if denom == 0:
        return False
    n_frac = n_count / denom
    if not (MIN_N_RATIO <= n_frac <= MAX_N_RATIO):
        return False

    # 2. Check maximum atom counts
    s_atoms = [a for a in atoms if a.GetSymbol() == "S"]
    if len(s_atoms) > MAX_COUNT_S: return False
    
    o_atoms = [a for a in atoms if a.GetSymbol() == "O"]
    if len(o_atoms) > MAX_COUNT_O: return False
    
    f_count = sum(1 for a in atoms if a.GetSymbol() == "F")
    if f_count > MAX_COUNT_F: return False
    
    cl_count = sum(1 for a in atoms if a.GetSymbol() == "Cl")
    if cl_count > MAX_COUNT_Cl: return False
    
    br_count = sum(1 for a in atoms if a.GetSymbol() == "Br")
    if br_count > MAX_COUNT_Br: return False

    # 3. Check sulfur (S) bonding state (valence)
    for s_atom in s_atoms:
        deg = s_atom.GetDegree()
        if deg not in (2, 6):
            return False

    return True

def mutate_atoms(mol, swaps=1, seed=None):
    rng = random.Random(seed)
    editable = Chem.RWMol(mol)
    
    allowed = MUTATE_ALLOWED
    weights = MUTATE_WEIGHTS
    pt = Chem.GetPeriodicTable()

    protected_indices = set()
    for smarts in PROTECTED_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                protected_indices.update(match)

    actual_swaps = rng.randint(1, swaps) if swaps > 1 else 1

    for _ in range(actual_swaps):
        atoms = list(editable.GetAtoms())
        
        candidates = [a for a in atoms if a.GetSymbol() != "O" and a.GetIdx() not in protected_indices]
        if not candidates:
            return mol

        atom = rng.choice(candidates)
        idx = atom.GetIdx()
        current_valence = sum([b.GetBondTypeAsDouble() for b in atom.GetBonds()])

        if current_valence > 1:
            allowed_filtered = [x for x in allowed if x not in ("F", "Cl", "Br")]
        else:
            allowed_filtered = allowed

        filtered_weights = [weights[allowed.index(x)] for x in allowed_filtered]
        if not allowed_filtered:
            continue

        new_symbol = rng.choices(allowed_filtered, weights=filtered_weights, k=1)[0]
        if new_symbol == atom.GetSymbol():
            continue

        editable.GetAtomWithIdx(idx).SetAtomicNum(pt.GetAtomicNumber(new_symbol))

    try:
        newmol = editable.GetMol()
        Chem.SanitizeMol(newmol)
        return newmol
    except Exception:
        return None

def run_aizynth(smiles, config, outfile, timeout_sec=600):
    log_file = str(outfile) + ".stderr.log"

    cmd = [
        "aizynthcli",
        "--smiles", smiles,
        "--config", str(config),
        "--output", str(outfile),
    ]

    try:
        subprocess.run(
            cmd, check=True,
            stdout=subprocess.DEVNULL, stderr=open(log_file, "wb"),
            timeout=timeout_sec
        )

        if not Path(outfile).exists():
            return {"ok": False, "error": f"No JSON output generated: {outfile}", "smiles": smiles}

        with open(outfile, "r", encoding="utf-8") as f:
            data = json.load(f)

        # --- Scan the entire JSON for is_solved==true nodes ---
        def collect_nodes(obj):
            if isinstance(obj, dict):
                yield obj
                for v in obj.values():
                    yield from collect_nodes(v)
            elif isinstance(obj, list):
                for v in obj:
                    yield from collect_nodes(v)

        solved_trees = []
        for node in collect_nodes(data):
            if not isinstance(node, dict):
                continue
            if node.get("metadata", {}).get("is_solved", False):
                scores = node.get("scores", {})
                summary = {
                    "is_solved": True,
                    "top_score": scores.get("state score", 0.0),
                    "number_of_steps": scores.get("number of reactions", 0),
                    "number_of_precursors": scores.get("number of pre-cursors", 0),
                    "number_of_precursors_in_stock": scores.get("number of pre-cursors in stock", 0),
                }
                solved_trees.append((summary, node))

        if solved_trees:
            # Sort by number_of_steps ascending
            solved_trees.sort(key=lambda x: x[0]["number_of_steps"])
            best_summary = solved_trees[0][0]
            trees = [t[1] for t in solved_trees]
            
            return {"ok": True, "summary": best_summary, "smiles": smiles, "trees": trees}
        else:
            return {"ok": True, "summary": {"is_solved": False}, "smiles": smiles}

    except Exception as e:
        return {"ok": False, "error": str(e), "smiles": smiles}



def _worker(args):
    smiles, config_path, tag = args
    outdir = Path("aizynth_runs")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / f"result_{tag}.json"
    return run_aizynth(smiles, config_path, outfile)


def _coerce_result_node(data):
    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            # Prioritize solved nodes
            for node in data["results"]:
                if node.get("is_solved", False):
                    return node
            return data["results"][0]
        return data

    if isinstance(data, list) and data and isinstance(data[0], dict):
        for node in data:
            if node.get("is_solved", False):
                return node
        return data[0]

    raise ValueError("Unknown AiZynthFinder output format")



# ---- Main processing ----
def safe_mutate(mol, swaps=1, seed=None):
    """Wrapper for mutate_atoms. Returns None on error."""
    try:
        return mutate_atoms(mol, swaps=swaps, seed=seed)
    except Exception:
        return None

def process_one(name, smiles, config, args):
    base_smi = canonical_smiles(smiles)
    if not base_smi:
        return {"ok": False, "error": "Invalid SMILES", "name": name, "smiles": smiles}

    print(f"[INFO] {name}: Original molecule {base_smi}")

    # Explore the original molecule first
    base_result = _worker((base_smi, config, f"{name}_0"))
    save_result(base_result, args.out)

    if base_result and base_result.get("summary", {}).get("is_solved"):
        print(f"[RESULT] {name}: Solved at original molecule!")
        return base_result

    # --- Generate mutants sequentially ---
    mol = Chem.MolFromSmiles(base_smi)
    rng = random.Random(args.seed)
    seen = set([base_smi])

    for i in range(args.samples):
        m2 = safe_mutate(mol, swaps=args.max_swaps, seed=rng.randint(0, 999999))
        if not m2:
            continue

        smi = Chem.MolToSmiles(m2, canonical=True)
        if smi in seen:
            continue
        seen.add(smi)

        # Constraint check
        if not check_constraints(m2):
            continue

        tag = f"{name}_{i+1}"
        print(f"[INFO] {name}: Testing mutant {tag} -> {smi}")
        result = _worker((smi, config, tag))
        save_result(result, args.out)

        if result and result.get("summary", {}).get("is_solved"):
            print(f"[RESULT] {name}: Solved at mutant {tag}!")
            #return result

    print(f"[RESULT] {name}: Not solved after {args.samples} attempts")
    return {"ok": True, "summary": {"is_solved": False}, "name": name, "smiles": base_smi}

def make_mutants_with_cn_ratio(smiles, n_samples=60, swaps=3, seed=0,
                               max_attempts_factor=1000,
                               max_swaps_limit=10):
    """
    Generate mutants while respecting the constraints in the configuration file.
    If n_samples is not reached even after the maximum number of attempts, increase max_swaps and try again.
    """
    mol = Chem.MolFromSmiles(smiles)
    rng = random.Random(seed)
    results = []
    seen = set()
    gen_stats = {}  # Record generation count for each swap count

    while len(results) < n_samples and swaps <= max_swaps_limit:
        attempts = 0
        max_attempts = n_samples * max_attempts_factor
        start_len = len(results)   # Record initial value per swap

        while len(results) < n_samples and attempts < max_attempts:
            attempts += 1
            m2 = mutate_atoms(mol, swaps=swaps, seed=rng.randint(0, 999999))
            if not m2:
                continue

            smi = Chem.MolToSmiles(m2, canonical=True)
            if smi in seen:
                continue

            if check_constraints(m2):
                seen.add(smi)
                results.append(smi)

        # Save statistics per stage
        added = len(results) - start_len
        gen_stats[swaps] = added
        print(f"[INFO] swaps={swaps}: generated {added} mutants "
              f"after {attempts} attempts (total {len(results)}/{n_samples})")

        if len(results) < n_samples:
            swaps += 1  # Increase mutation intensity and try again
        else:
            break

    # --- Output final statistics ---
    print("\n=== Mutation summary by swaps ===")
    for s, count in gen_stats.items():
        print(f"  swaps={s}: {count} mutants generated")

    if len(results) < n_samples:
        print(f"[FINAL WARN] Only generated {len(results)} mutants even at swaps={swaps-1}")

    return results


def _worker_skip(args):
    smiles, config_path, tag, name, solved_dict = args

    # Skip if already solved
    if solved_dict.get(name, False):
        return {"ok": True, "skipped": True, "smiles": smiles, "tag": tag, "name": name}

    with tempfile.TemporaryDirectory() as td:
        result = run_aizynth(smiles, config_path, Path(td) / f"result_{tag}.json")
        if result is None:
            return {"ok": False, "error": "run_aizynth returned None",
                    "smiles": smiles, "tag": tag, "name": name}
        result["tag"] = tag
        result["name"] = name

        # If solved, record in shared dictionary (enable the following if you want to skip once an input is successfully solved)
        #if result.get("summary", {}).get("is_solved", False):
        #    solved_dict[name] = True

        return result

def main():
    ap = argparse.ArgumentParser(description="Run AiZynthFinder in batch")
    ap.add_argument("--ini", default="rss_config.ini", help="Configuration INI file")
    ap.add_argument("--smiles", default=EXEC_SMILES, help="Single SMILES")
    ap.add_argument("--batch", default=EXEC_BATCH, help="SMILES list file (TSV format: name\\tSMILES)")
    ap.add_argument("--config", default=EXEC_CONFIG, help="AiZynthFinder config.yml")
    ap.add_argument("--samples", type=int, default=EXEC_SAMPLES, help="Number of candidates to generate")
    ap.add_argument("--max_swaps", type=int, default=EXEC_MAX_SWAPS, help="Maximum number of substitutions per molecule")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=EXEC_JOBS, help="Number of parallel jobs")
    ap.add_argument("--out", default=EXEC_OUT, help="Result save destination (JSONL)")
    args = ap.parse_args()

    if not args.config:
        ap.error("--config or [Execution] config in the INI file is required")

    # Create the parent directory to store the results
    base_out_dir = Path("rss_results")
    base_out_dir.mkdir(exist_ok=True)

    # Note: The traditional args.out is not used, so we deleted/ignored it, but kept it from creating an empty file just in case.

    # Read inputs
    inputs = []
    if args.smiles:
        inputs = [("input", args.smiles)]
    elif args.batch:
        with open(args.batch) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                name, smi = line.split(None, 1)
                inputs.append((name, smi))

    manager = Manager()
    solved_dict = manager.dict()

    # Process each molecule and summarize the results
    summary_rows = []
    
    for name, smi in inputs:
        base_smi = canonical_smiles(smi)
        if not base_smi:
            continue

        # Create a directory for each ligand under rss_results
        work_dir = base_out_dir / name
        work_dir.mkdir(exist_ok=True)
        ligand_out = work_dir / "mutants_results.jsonl"
        ligand_sorted_out = work_dir / "mutants_results_successful_sorted.jsonl"

        # Initialize the output file
        open(ligand_out, "w").close()

        def save_result(r):
            import json
            with open(ligand_out, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")

        # --- 1) First run the original synchronously ---
        print(f"\n[INFO] Processing original molecule: {name}")
        base_result = _worker_skip((base_smi, args.config, f"{name}_0", name, solved_dict))
        save_result(base_result)

        # --- 2) If the original is not solved (or even if it is), create mutants and run them in parallel ---
        mutants = make_mutants_with_cn_ratio(
            base_smi,
            n_samples=args.samples,
            swaps=args.max_swaps,
            seed=args.seed,
            max_swaps_limit=10
        )
        tasks = [(msmi, args.config, f"{name}_{i}", name, solved_dict)
                 for i, msmi in enumerate(mutants, 1)]

        if not tasks:
            print(f"[WARN] No valid mutants generated for {name}")
        else:
            print(f"[INFO] {name}: running {len(tasks)} mutants in parallel")
            with Pool(processes=args.jobs) as pool:
                for r in pool.imap_unordered(_worker_skip, tasks):
                    save_result(r)

        # --- 3) Extract successful molecules, sort, and create summary for each ligand ---
        best_steps = None
        if os.path.exists(ligand_out):
            filtered = []
            with open(ligand_out, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    
                    # Extract successful cases only
                    if d.get("ok") and d.get("summary", {}).get("is_solved"):
                        filtered.append(d)
                        steps = d["summary"].get("number_of_steps")
                        if steps is not None:
                            best_steps = steps if best_steps is None else min(best_steps, steps)

            # Sort in ascending order of number_of_steps
            filtered.sort(key=lambda x: x["summary"].get("number_of_steps", 9999))

            # --- Output list (CSV) and synthesis routes (PNG) of all successful mutants ---
            if filtered:
                ligand_csv_out = work_dir / "mutants_summary.csv"
                print(f"🧩 {name}: Generating PNGs and CSV list for all successful mutants ({len(filtered)} items)...")
                try:
                    from aizynthfinder.reactiontree import ReactionTree
                    import csv
                    
                    total_images = 0
                    with open(ligand_csv_out, "w", newline="", encoding="utf-8") as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["rank", "ligand_name", "smiles", "steps", "png_files"])
                        
                        for rank, mutant in enumerate(filtered):
                            trees = mutant.get("trees")
                            tag = mutant.get("tag", f"rank{rank:03d}")
                            steps = mutant.get("summary", {}).get("number_of_steps", "")
                            smi = mutant.get("smiles", "")
                            
                            saved_pngs = []
                            if trees:
                                for itree, tree in enumerate(trees):
                                    png_name = f"rank{rank:03d}_{tag}_route{itree:03d}.png"
                                    img = ReactionTree.from_dict(tree).to_image()
                                    img.save(work_dir / png_name)
                                    saved_pngs.append(png_name)
                                    total_images += 1
                            
                            writer.writerow([rank+1, name, smi, steps, ", ".join(saved_pngs)])

                    print(f"✅ {name}: List saved to {ligand_csv_out.name}, a total of {total_images} route images output")
                except Exception as e:
                    print(f"❌ Failed to generate images/CSV for {name}: {e}")
        else:
            print(f"⚠️ Result file not found: {ligand_out}")

        if best_steps is None:
            print(f"[WARN] No successful molecules in {name}")
            summary_rows.append({
                "name": name,
                "solved_count": 0,
                "min_steps": "",
                "best_smiles": ""
            })
        else:
            best_mutant = filtered[0]
            summary_rows.append({
                "name": name, 
                "solved_count": len(filtered), 
                "min_steps": best_steps, 
                "best_smiles": best_mutant.get("smiles", "")
            })

    # --- 4) Write the summary CSV ---
    summary_file = base_out_dir / "summary.csv"
    import csv
    with open(summary_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["name", "solved_count", "min_steps", "best_smiles"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n✅ Summary written to {summary_file}")
    print(f"✅ Total ligands successfully processed: {len(summary_rows)}")

if __name__ == "__main__":
    main()
