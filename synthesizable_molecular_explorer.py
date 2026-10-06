#!/usr/bin/env python3
"""
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
RDLogger.DisableLog('rdApp.*')

# ---- User settings (Read from external file) ----
CONFIG_FILE = "sme_config.ini"
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
EXEC_SAMPLES = 50
EXEC_MAX_SWAPS = 10
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

def mutate_atoms(mol, swaps=1, seed=None):
    rng = random.Random(seed)
    editable = Chem.RWMol(mol)

    allowed = MUTATE_ALLOWED
    weights = MUTATE_WEIGHTS
    pt = Chem.GetPeriodicTable()

    # Extract indices of atoms to protect
    protected_indices = set()
    for smarts in PROTECTED_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                protected_indices.update(match)

    # Execute a random number of times within the specified swaps range from 1
    actual_swaps = rng.randint(1, swaps) if swaps > 1 else 1

    for _ in range(actual_swaps):
        # Get the latest atom list for each swap
        atoms = list(editable.GetAtoms())
        
        # Exclude O atoms and protected atoms when selecting mutation targets
        candidates = [a for a in atoms if a.GetSymbol() != "O" and a.GetIdx() not in protected_indices]
        if not candidates:
            return mol  # Return as is if no candidates

        # Randomly select a target from candidates
        atom = rng.choice(candidates)
        idx = atom.GetIdx()
        current_valence = sum([b.GetBondTypeAsDouble() for b in atom.GetBonds()])

        # Halogens are only allowed on single bonds (sp3 terminals)
        if current_valence > 1:
            allowed_filtered = [x for x in allowed if x not in ("F", "Cl", "Br")]
        else:
            allowed_filtered = allowed

        # Adjust the length of weights to match the filtered elements
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

def check_constraints(mol):
    """Check the constraints that mutants must satisfy"""
    num_C = num_N = num_O = num_S = num_F = num_Cl = num_Br = 0

    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym == "C": num_C += 1
        elif sym == "N": num_N += 1
        elif sym == "O": num_O += 1
        elif sym == "S":
            num_S += 1
            # S is invalid unless the number of bonds is exactly 2
            val = sum([b.GetBondTypeAsDouble() for b in atom.GetBonds()])
            if val != 2:
                return False
        elif sym == "Br": num_Br += 1
        elif sym == "Cl": num_Cl += 1
        elif sym == "F": num_F += 1

    # --- Check CN ratio ---
    total_CN = num_C + num_N
    if total_CN > 0:
        ratio_N = num_N / total_CN
        if ratio_N < MIN_N_RATIO or ratio_N > MAX_N_RATIO:
            return False

    # --- Limit counts ---
    if num_S > MAX_COUNT_S:
        return False
    if num_O > MAX_COUNT_O:
        return False
    if num_Br > MAX_COUNT_Br:
        return False
    if num_Cl > MAX_COUNT_Cl:
        return False
    if num_F > MAX_COUNT_F:
        return False

    return True


def make_mutants_general(smiles, n_samples=50, swaps=1, seed=0):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    rng = random.Random(seed)
    seen = set()
    results = []

    while len(results) < n_samples:
        m2 = mutate_atoms(mol, swaps=swaps, seed=rng.randint(0, 999999))
        if not m2:
            continue
        smi = Chem.MolToSmiles(m2, canonical=True)
        if smi not in seen:
            seen.add(smi)
            results.append(smi)

    return results

def mutate_worker_batch(args):
    mol, max_swaps, seed, n = args
    rng = random.Random(seed)
    results = []
    attempts = 0

    for _ in range(n):
        m2 = safe_mutate(mol, swaps=max_swaps, seed=rng.randint(0, 999999))
        if not m2:
            continue
        if not check_constraints(m2):
            continue
        results.append(Chem.MolToSmiles(m2, canonical=True))

    return results, attempts

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
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
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

    except subprocess.TimeoutExpired:
        # Timeout specific handling
        return {"ok": False, "error": f"timeout after {timeout_sec}s", "smiles": smiles}

    except Exception as e:
        print(f"[ERROR] AiZynthFinder failed: {e}")
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
    """Process a single molecule (analyze mutants in parallel)"""

    base_smi = canonical_smiles(smiles)
    if not base_smi:
        return {"ok": False, "error": "Invalid SMILES", "name": name, "smiles": smiles}

    print(f"[INFO] {name}: Original molecule {base_smi}")

    # --- First analyze the original ---
    base_result = _worker((base_smi, config, f"{name}_0"))
    base_result["name"] = name
    if base_result and base_result.get("summary", {}).get("is_solved"):
        print(f"[RESULT] {name}: Solved at original molecule!")
        return base_result

    # --- Generate mutants ---
    mol = Chem.MolFromSmiles(base_smi)
    rng = random.Random(args.seed)
    seen = set([base_smi])
    mutants = []

    max_attempts = args.samples * 10000
    attempts = 0

    print(f"[INFO] {name}: Generating mutants in parallel...")

    with Pool(processes=args.jobs) as pool:

        per_worker = 30  
        batch_size = args.jobs * 4  
        done = False      

        while len(mutants) < args.samples and attempts < max_attempts:

            tasks = [(mol, args.max_swaps, rng.randint(0, 999999), per_worker)
                    for _ in range(batch_size)]

            for smi_list, n_attempts in pool.imap_unordered(mutate_worker_batch, tasks):
                attempts += per_worker            
                       
                for smi in smi_list:
                    if smi and smi not in seen:
                        seen.add(smi)
                        mutants.append(smi)

                        if len(mutants) >= args.samples:
                            pool.terminate()
                            done = True
                            break
            
                if done:
                    break

        if attempts >= max_attempts:
            print(f"[WARN] {name}: Mutation attempts exceeded {max_attempts}, got {len(mutants)} mutants")

    # --- Analyze mutants in parallel ---
    print(f"[INFO] {name}: Running AiZynthFinder on {len(mutants)} mutants with {args.jobs} CPUs")

    tasks = [(smi, config, f"{name}_{i+1}") for i, smi in enumerate(mutants)]
    results = []

    with Pool(processes=args.jobs) as pool:
        for res in pool.imap_unordered(_worker, tasks):
            res["name"] = name
            results.append(res)
            if res.get("summary", {}).get("is_solved"):
                print(f"[RESULT] {name}: Solved at mutant {res['smiles']}")
                pool.terminate()  # Immediately terminate other tasks
                return res

    # --- All failed ---
    print(f"[RESULT] {name}: Not solved after {len(mutants)} mutants.")
    return {"ok": True, "summary": {"is_solved": False}, "name": name, "smiles": base_smi}




def make_mutants_with_cn_ratio(smiles, n_samples=50, swaps=1, seed=0,
                               min_n_frac=0.1, max_n_frac=0.4,
                               max_sulfur=1):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    rng = random.Random(seed)
    seen, results = set(), []

    while len(results) < n_samples:
        m2 = mutate_atoms(mol, swaps=swaps, seed=rng.randint(0, 999999))
        if not m2:
            continue

        smi = Chem.MolToSmiles(m2, canonical=True)
        if smi in seen:
            continue

        atoms = m2.GetAtoms()

        # --- N ratio (Denominator is C+N) ---
        c_count = sum(1 for a in atoms if a.GetSymbol() == "C")
        n_count = sum(1 for a in atoms if a.GetSymbol() == "N")
        denom = c_count + n_count
        n_frac = n_count / denom if denom > 0 else 0.0

        # --- S conditions ---
        s_atoms = [a for a in atoms if a.GetSymbol() == "S"]
        if len(s_atoms) > max_sulfur:
            continue
        if len(s_atoms) == 1:
            deg = s_atoms[0].GetDegree()
            if deg not in (2, 6):   # S is only allowed as 2-coordinate or 6-coordinate
                continue

        # --- O conditions (max 1 aromatic ring O) ---
        aromatic_oxygens = [a for a in atoms if a.GetSymbol() == "O"
                            and a.GetIsAromatic() and a.IsInRing()]
        if len(aromatic_oxygens) > 1:
            continue      

        # --- Br limits ---
        br_count = sum(1 for a in atoms if a.GetSymbol() == "Br")
        if br_count > 1:
            continue  

        # --- Final conditions ---
        if min_n_frac <= n_frac <= max_n_frac:
            seen.add(smi)
            results.append(smi)

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

        # Record in shared dict if solved
        if result.get("summary", {}).get("is_solved", False):
            solved_dict[name] = True

        return result

# ---- Globally defined ----
def worker_wrapper(inp_args):
    """Wrapper for multiprocessing"""
    name, smi, config, args = inp_args
    return process_one(name, smi, config, args)

def main():
    ap = argparse.ArgumentParser(description="AiZynthFinder batch runner with mutant parallelization")
    ap.add_argument("--ini", default="sme_config.ini", help="Configuration INI file")
    ap.add_argument("--smiles", default=EXEC_SMILES, help="Single SMILES")
    ap.add_argument("--batch", default=EXEC_BATCH, help="SMILES list file (TSV format: name\\tSMILES)")
    ap.add_argument("--config", default=EXEC_CONFIG, help="AiZynthFinder config.yml")
    ap.add_argument("--samples", type=int, default=EXEC_SAMPLES, help="Maximum number of attempts (number of mutants)")
    ap.add_argument("--max_swaps", type=int, default=EXEC_MAX_SWAPS, help="Maximum number of substitutions per molecule")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=EXEC_JOBS, help="Number of parallel jobs (per mutant)")
    ap.add_argument("--out", default=EXEC_OUT, help="Result save destination (JSONL)")
    args = ap.parse_args()
    
    if not args.config:
        ap.error("--config or [Execution] config in the INI file is required")

    # Initialize output file
    open(args.out, "w").close()

    # Read input
    inputs = []
    if args.smiles:
        inputs = [("input", args.smiles)]
    elif args.batch:
        with open(args.batch) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) == 1:
                    # IDが無くSMILESのみの場合は自動で名前を付ける
                    name = f"mol_{len(inputs) + 1}"
                    smi = parts[0]
                else:
                    # "ID SMILES" と "SMILES ID" の両方に対応
                    # 記号を含むなどSMILESらしい文字列を判定
                    if any(c in parts[0] for c in "=#()@[]") or len(parts[0]) > len(parts[1]):
                        smi, name = parts[0], parts[1]
                    else:
                        name, smi = parts[0], parts[1]
                inputs.append((name, smi))

    print(f"[INFO] Loaded {len(inputs)} molecules. Each molecule will use up to {args.jobs} CPUs.")

    summary_rows = []

    # --- Process each molecule sequentially ---
    for name, smi in inputs:
        print(f"\n[START] Processing {name}")
        result = process_one(name, smi, args.config, args)

        # Write to JSONL (exclude huge tree data to make it easier to read)
        import copy
        result_to_save = copy.deepcopy(result)
        result_to_save.pop("trees", None)
        with open(args.out, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(result_to_save, ensure_ascii=False) + "\n")
        
        is_solved = result.get("summary", {}).get("is_solved", False)
        steps = result.get("summary", {}).get("number_of_steps", "")
        summary_rows.append({
            "name": name,
            "smiles": result.get("smiles", smi),
            "is_solved": is_solved,
            "steps": steps
        })

        if result.get("summary", {}).get("is_solved"):
            print(f"[SOLVED] {name} ✅")
            
            # --- Generate image (PNG) ---
            trees = result.get("trees")
            if trees:
                work_dir = Path("sme_results") / name
                work_dir.mkdir(parents=True, exist_ok=True)
                print(f"🧩 {name}: Generating PNGs...")
                try:
                    from aizynthfinder.reactiontree import ReactionTree
                    for itree, tree in enumerate(trees):
                        img = ReactionTree.from_dict(tree).to_image()
                        img.save(work_dir / f"route{itree:03d}.png")
                    print(f"✅ {name}: {len(trees)} route images saved to {work_dir}/")
                except Exception as e:
                    print(f"❌ Failed to generate images for {name}: {e}")
            else:
                print(f"⚠ {name}: Skipping image generation because tree data is not included in JSON")
        else:
            print(f"[FAILED] {name} ❌")

    # --- Write summary CSV ---
    base_out_dir = Path("sme_results")
    base_out_dir.mkdir(exist_ok=True)
    summary_file = base_out_dir / "summary.csv"
    import csv
    with open(summary_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["name", "smiles", "is_solved", "steps"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n[INFO] All {len(inputs)} molecules processed. Results saved to {args.out}")
    print(f"✅ Summary written to {summary_file}")


if __name__ == "__main__":
    main()
