// mpas_li_rf_predict.cpp
//
// Thin extern "C" wrapper around rf_predict_core.h, exposed to Fortran via
// iso_c_binding from li_velocity_gnn (mpas_li_velocity_gnn.F). Used for the
// random-forest error check that decides whether a given GNN velocity solve
// should be trusted or whether MALI should fall back to the external (FO)
// dycore for that solve.
//
// rf_predict_core.h is mirrored verbatim from the offline RF exporter
// (rf_export/rf_predict_core.h, alongside rf_predict.cpp, the CLI used to
// develop/validate these forests against ranger). Keep the two copies
// identical so in-MALI and offline predictions never diverge.
//
// This file intentionally never throws across the Fortran/C++ boundary:
// every entry point catches std::exception and returns an error code,
// stashing the message for li_rf_last_error().

#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "rf_predict_core.h"

namespace {

using rfcore::Forest;
using rfcore::LevelMapping;

// Only two forests are used today (x-velocity error, y-velocity error), but
// size this generously; slot indices are caller-supplied small integers.
constexpr int kMaxSlots = 8;

struct Slot {
  bool loaded = false;
  Forest forest;
  LevelMapping mapping;
};

Slot g_slots[kMaxSlots];
std::string g_last_error;

// The GNN velocity emulator's 12 input features, in the fixed order
// produced by li_velocity_gnn_solve (mpas_li_velocity_gnn.F). The random
// forests must have been trained on exactly this feature set, in this
// order; li_rf_load() checks this at load time rather than trusting
// position blindly.
const char* const kExpectedFeatureNames[12] = {
    "xcell", "ycell", "areacell", "thickness", "bedtop", "log_fric",
    "basal_temp", "mean_temp", "stiff", "float_thick", "ground_mask",
    "boundary_mask"};
constexpr int kExpectedNumFeatures = 12;

bool slot_in_range(int slot) { return slot >= 0 && slot < kMaxSlots; }

// Draw `nSample` distinct indices from [0, nRows) using selection sampling
// (a partial Fisher-Yates shuffle). Deterministic given (nRows, nSample,
// seed), independent of how many ranks the caller runs on, provided the
// caller passes rows in a decomposition-independent order (global cell ID
// order, as li_velocity_gnn_solve does).
std::vector<size_t> sample_indices(size_t nRows, size_t nSample, uint64_t seed) {
  std::vector<size_t> pool(nRows);
  for (size_t i = 0; i < nRows; ++i) pool[i] = i;
  nSample = std::min(nSample, nRows);
  std::mt19937_64 rng(seed);
  for (size_t i = 0; i < nSample; ++i) {
    std::uniform_int_distribution<size_t> dist(i, nRows - 1);
    size_t j = dist(rng);
    std::swap(pool[i], pool[j]);
  }
  pool.resize(nSample);
  return pool;
}

}  // namespace

extern "C" {

// Load a forest into `slot` from `path` (expects `path`.features.txt and,
// if the forest has factor predictors, `path`.factor_levels.tsv alongside
// it). Returns 0 on success, nonzero on failure (see li_rf_last_error()).
int li_rf_load(int slot, const char* path) {
  if (!slot_in_range(slot)) { g_last_error = "li_rf_load: slot out of range"; return 1; }
  try {
    std::string p(path);
    Forest f = rfcore::load_forest(p);
    if (int(f.names.size()) != kExpectedNumFeatures) {
      g_last_error = "li_rf_load: expected " + std::to_string(kExpectedNumFeatures) +
                      " features, forest has " + std::to_string(f.names.size());
      return 1;
    }
    for (int i = 0; i < kExpectedNumFeatures; ++i) {
      if (f.names[size_t(i)] != kExpectedFeatureNames[i]) {
        g_last_error = "li_rf_load: feature " + std::to_string(i) + " is '" + f.names[size_t(i)] +
                        "', expected '" + std::string(kExpectedFeatureNames[i]) +
                        "' (forest feature order must match the GNN input feature order)";
        return 1;
      }
    }
    LevelMapping mapping = rfcore::read_factor_levels(p + ".factor_levels.tsv", f, false);
    g_slots[slot].forest = std::move(f);
    g_slots[slot].mapping = std::move(mapping);
    g_slots[slot].loaded = true;
    return 0;
  } catch (const std::exception& e) {
    g_last_error = std::string("li_rf_load: ") + e.what();
    return 1;
  }
}

// Number of predictors in the loaded forest in `slot`, or -1 if `slot` has
// not been loaded. Callers use this to sanity-check against their own
// feature count before calling li_rf_count_ones().
int li_rf_num_features(int slot) {
  if (!slot_in_range(slot) || !g_slots[slot].loaded) return -1;
  return int(g_slots[slot].forest.names.size());
}

// Draw a deterministic random subsample of `nSample` rows (without
// replacement, out of `nRows`) from the column-major feature matrix `x`
// (Fortran layout: x[row + col*nRows], nRows-by-nFeat) and run the forest
// loaded in `slot` on each sampled row. Returns 0 on success and sets
// *nOnes to the number of rows predicted class 1 and *nTies to the number
// of exactly-tied votes (each tie is still counted into *nOnes or not via a
// coin flip, same as the offline CLI's default --tie random policy, using
// `seed` to drive both the sampling and the tie-break RNG).
//
// nFeat must equal li_rf_num_features(slot); this is checked defensively.
int li_rf_count_ones(int slot, const double* x, int nRows, int nFeat, int nSample,
                      uint64_t seed, int* nOnes, int* nTies) {
  if (!slot_in_range(slot) || !g_slots[slot].loaded) {
    g_last_error = "li_rf_count_ones: slot not loaded";
    return 1;
  }
  if (nRows < 0 || nFeat < 0 || nSample < 0 || !x || !nOnes || !nTies) {
    g_last_error = "li_rf_count_ones: invalid arguments";
    return 1;
  }
  const Forest& f = g_slots[slot].forest;
  const LevelMapping& mapping = g_slots[slot].mapping;
  if (size_t(nFeat) != f.names.size()) {
    g_last_error = "li_rf_count_ones: caller nFeat=" + std::to_string(nFeat) +
                    " does not match forest width=" + std::to_string(f.names.size());
    return 1;
  }
  try {
    *nOnes = 0;
    *nTies = 0;
    if (nRows == 0 || nSample == 0) return 0;
    // One RNG stream drives both row sampling and tie-breaking; offsetting
    // the tie-break seed keeps the two draws independent.
    std::vector<size_t> rows = sample_indices(size_t(nRows), size_t(nSample), seed);
    std::mt19937_64 tie_rng(seed ^ 0x9E3779B97F4A7C15ULL);
    std::vector<double> xrow(size_t(nFeat), 0.0);
    for (size_t row : rows) {
      for (int c = 0; c < nFeat; ++c) {
        double raw = x[row + size_t(c) * size_t(nRows)];
        xrow[size_t(c)] = mapping.levels[size_t(c)].empty()
                               ? raw
                               : rfcore::factor_code_numeric(raw, mapping.levels[size_t(c)], f.names[size_t(c)]);
      }
      bool is_tie = false;
      int pred = rfcore::forest_vote(f, xrow, &is_tie);
      if (is_tie) {
        ++(*nTies);
        pred = int(tie_rng() & 1ULL);
      }
      *nOnes += pred;
    }
    return 0;
  } catch (const std::exception& e) {
    g_last_error = std::string("li_rf_count_ones: ") + e.what();
    return 1;
  }
}

// Release the forest loaded in `slot`, if any. Safe to call on an unloaded
// or out-of-range slot.
void li_rf_free(int slot) {
  if (!slot_in_range(slot)) return;
  g_slots[slot] = Slot();
}

// Copy the message describing the most recent failure from any of the
// functions above into the caller-supplied buffer `buf` of length `buflen`
// bytes, truncating if necessary and always NUL-terminating. A fixed-size
// buffer (rather than returning a `const char*`) keeps this call simple on
// the Fortran/iso_c_binding side.
void li_rf_last_error(char* buf, int buflen) {
  if (!buf || buflen <= 0) return;
  size_t n = std::min(size_t(buflen - 1), g_last_error.size());
  std::memcpy(buf, g_last_error.data(), n);
  buf[n] = '\0';
}

}  // extern "C"
