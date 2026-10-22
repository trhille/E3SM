// ranger 0.18.0 forest predictor with ranger-compatible factor-level conversion.
// RGR01801 binary format (unchanged from earlier export_ranger018.R).
// Loads a separate factor-level TSV next to weights:
//      <weights>.factor_levels.tsv
// Each line: feature_name<TAB>level_1<TAB>level_2[<TAB>level_3...]
// The CSV values are labels; like ranger's predict.R, these must be converted
// into 1-based integer factor codes before evaluating any splits.
//
// This header is the shared core used both by the standalone rf_predict CLI
// (rf_export/rf_predict_main.cpp) and by the MALI-embedded wrapper
// (components/mpas-albany-landice/src/mode_forward/mpas_li_rf_predict.cpp).
// Keep the two copies of this file identical so online (in-MALI) and offline
// predictions are guaranteed to agree.
#ifndef RF_PREDICT_CORE_H
#define RF_PREDICT_CORE_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace rfcore {

struct Node { int32_t left, right, missing, feature, ordered, leaf; double value; };
struct Tree { std::vector<Node> nodes; };
struct Forest { std::vector<std::string> names; std::vector<Tree> trees; };

// Model metadata is a map from predictor position to the *training* levels.
// For any factor (even when ranger marks it is.ordered = TRUE), predict.R
// turns a label 0/1 into factor codes 1/2 through data.matrix().
struct LevelMapping {
  std::vector<std::vector<std::string>> levels;
  size_t feature_count = 0;
};

inline std::vector<std::string> split(const std::string& line, char sep) {
  std::vector<std::string> fields;
  size_t start = 0;
  for (;;) {
    size_t end = line.find(sep, start);
    std::string s = line.substr(start, end == std::string::npos ? std::string::npos : end - start);
    if (!s.empty() && s.back() == '\r') s.pop_back();
    fields.push_back(s);
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return fields;
}

inline double parse_number(const std::string& s, const std::string& context) {
  char* end = nullptr;
  double v = std::strtod(s.c_str(), &end);
  if (s.empty() || end != s.c_str() + s.size() || !std::isfinite(v))
    throw std::runtime_error("Invalid numerical input for " + context + ": '" + s + "'");
  return v;
}

// Decode a little-endian 32-bit integer from a raw buffer.
inline int32_t decode_i32(const unsigned char* b) {
  return static_cast<int32_t>(uint32_t(b[0]) | (uint32_t(b[1]) << 8) |
                               (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24));
}

// Decode a little-endian 64-bit IEEE double from a raw buffer.
inline double decode_f64(const unsigned char* b) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= uint64_t(b[i]) << (8 * i);
  double d;
  static_assert(sizeof d == sizeof v, "64-bit double required");
  std::memcpy(&d, &v, 8);
  return d;
}

inline int32_t read_i(std::istream& in) {
  unsigned char b[4];
  if (!in.read(reinterpret_cast<char*>(b), 4)) throw std::runtime_error("Truncated weights (integer)");
  return decode_i32(b);
}

// Load a forest from `path`, reading each tree's node block in a single
// buffered read rather than one istream::read per field (meaningfully faster
// for forests with tens of millions of nodes).
inline Forest load_forest(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open weights: " + path);
  char magic[8];
  if (!in.read(magic, 8) || std::string(magic, 8) != "RGR01801")
    throw std::runtime_error("Wrong weights format (expected RGR01801): " + path);
  int p = read_i(in), nt = read_i(in);
  if (p < 1 || p > 100000 || nt < 1 || nt > 1000000) throw std::runtime_error("Invalid forest dimensions");
  std::ifstream side(path + ".features.txt");
  if (!side) throw std::runtime_error("Missing feature names file: " + path + ".features.txt");
  Forest f;
  std::string name;
  while (std::getline(side, name)) {
    if (!name.empty() && name.back() == '\r') name.pop_back();
    f.names.push_back(name);
  }
  if (int(f.names.size()) != p) throw std::runtime_error("Number of feature names differs from forest width");

  // Each node is 6 int32 fields (left,right,missing,feature,ordered,leaf) + 1 double = 32 bytes.
  const size_t kNodeBytes = 4 * 6 + 8;
  std::vector<unsigned char> buf;
  f.trees.reserve(nt);
  for (int t = 0; t < nt; ++t) {
    int n = read_i(in);
    if (n < 1 || n > 100000000) throw std::runtime_error("Invalid number of nodes");
    buf.resize(size_t(n) * kNodeBytes);
    if (!in.read(reinterpret_cast<char*>(buf.data()), std::streamsize(buf.size())))
      throw std::runtime_error("Truncated weights (tree node block)");
    Tree tr;
    tr.nodes.resize(size_t(n));
    for (int i = 0; i < n; ++i) {
      const unsigned char* b = buf.data() + size_t(i) * kNodeBytes;
      Node node;
      node.left = decode_i32(b + 0);
      node.right = decode_i32(b + 4);
      node.missing = decode_i32(b + 8);
      node.feature = decode_i32(b + 12);
      node.ordered = decode_i32(b + 16);
      node.leaf = decode_i32(b + 20);
      node.value = decode_f64(b + 24);
      if (node.leaf) {
        if (node.value != 0.0 && node.value != 1.0) throw std::runtime_error("Invalid terminal class in weights");
      } else if (node.feature < 0 || node.feature >= p || node.left < 0 || node.right < 0 ||
                 node.left >= n || node.right >= n) {
        throw std::runtime_error("Invalid split node");
      }
      tr.nodes[size_t(i)] = node;
    }
    f.trees.push_back(std::move(tr));
  }
  return f;
}

inline LevelMapping read_factor_levels(const std::string& path, const Forest& f, bool required) {
  std::ifstream in(path);
  if (!in) {
    if (required) throw std::runtime_error("Cannot open factor-level metadata: " + path);
    return {std::vector<std::vector<std::string>>(f.names.size()), 0};
  }
  LevelMapping out;
  out.levels.resize(f.names.size());
  std::string line;
  size_t lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    if (line.empty() || line[0] == '#') continue;
    auto fields = split(line, '\t');
    if (fields.size() < 3) throw std::runtime_error("Bad factor-level metadata at line " + std::to_string(lineno));
    auto it = std::find(f.names.begin(), f.names.end(), fields[0]);
    if (it == f.names.end()) throw std::runtime_error("Unknown factor predictor in metadata: " + fields[0]);
    size_t ix = size_t(it - f.names.begin());
    if (!out.levels[ix].empty()) throw std::runtime_error("Duplicate factor name in metadata: " + fields[0]);
    out.levels[ix].assign(fields.begin() + 1, fields.end());
    for (const auto& level : out.levels[ix]) {
      if (level.empty()) throw std::runtime_error("Empty factor level in " + fields[0]);
    }
    out.feature_count++;
  }
  return out;
}

// Convert a raw string label to its 1-based factor code.
inline double factor_code(const std::string& raw, const std::vector<std::string>& levels,
                           const std::string& name) {
  for (size_t i = 0; i < levels.size(); ++i) if (raw == levels[i]) return double(i + 1);
  double v = parse_number(raw, name);
  for (size_t i = 0; i < levels.size(); ++i) {
    char* end = nullptr;
    double level = std::strtod(levels[i].c_str(), &end);
    if (!levels[i].empty() && end == levels[i].c_str() + levels[i].size() && std::isfinite(level) && level == v)
      return double(i + 1);
  }
  throw std::runtime_error("Unknown factor level '" + raw + "' for predictor " + name);
}

// Convert an already-numeric 0/1 value to its 1-based factor code, for
// callers (like MALI) that pass numeric mask values directly rather than
// string labels.
inline double factor_code_numeric(double v, const std::vector<std::string>& levels,
                                   const std::string& name) {
  for (size_t i = 0; i < levels.size(); ++i) {
    char* end = nullptr;
    double level = std::strtod(levels[i].c_str(), &end);
    if (!levels[i].empty() && end == levels[i].c_str() + levels[i].size() && std::isfinite(level) && level == v)
      return double(i + 1);
  }
  throw std::runtime_error("Unknown factor level '" + std::to_string(v) + "' for predictor " + name);
}

inline int tree_vote(const Tree& tr, const std::vector<double>& x) {
  size_t node_id = 0;
  for (size_t step = 0; step < tr.nodes.size(); ++step) {
    const Node& node = tr.nodes[node_id];
    if (node.leaf) return int(node.value);
    double value = x.at(size_t(node.feature));
    int child = 0;
    if (node.ordered) {
      if (std::isnan(value)) child = node.missing > 0 ? node.missing : node.left;
      else child = value <= node.value ? node.left : node.right;
    } else {
      if (!std::isfinite(value) || value < 1 || value > 64 || std::floor(value) != value)
        throw std::runtime_error("Invalid categorical factor code");
      uint64_t mask = uint64_t(std::floor(node.value));
      child = (mask & (uint64_t(1) << (unsigned(value) - 1))) ? node.right : node.left;
    }
    if (child < 0 || size_t(child) >= tr.nodes.size()) throw std::runtime_error("Child node out of range");
    node_id = size_t(child);
  }
  throw std::runtime_error("Cycle in forest");
}

// Run the whole forest on one row and return 0/1. `*is_tie` is set if the
// vote was exactly tied (caller decides how to break it).
inline int forest_vote(const Forest& f, const std::vector<double>& x, bool* is_tie) {
  size_t positive = 0;
  for (const auto& tree : f.trees) positive += size_t(tree_vote(tree, x));
  *is_tie = (2 * positive == f.trees.size());
  return 2 * positive > f.trees.size();
}

}  // namespace rfcore

#endif  // RF_PREDICT_CORE_H
