#include <algorithm>
#include <bitset>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct CompileError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

std::string trim(std::string s) {
    auto non_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), non_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), non_space).base(), s.end());
    return s;
}

std::string strip_comments(const std::string &input) {
    std::string out;
    enum class State { Normal, Line, Block } state = State::Normal;
    for (std::size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        char n = i + 1 < input.size() ? input[i + 1] : '\0';
        if (state == State::Normal && c == '/' && n == '/') {
            state = State::Line;
            ++i;
        } else if (state == State::Normal && c == '/' && n == '*') {
            state = State::Block;
            ++i;
        } else if (state == State::Line && c == '\n') {
            out.push_back(c);
            state = State::Normal;
        } else if (state == State::Block && c == '*' && n == '/') {
            state = State::Normal;
            ++i;
        } else if (state == State::Normal) {
            out.push_back(c);
        }
    }
    if (state == State::Block) throw CompileError("unterminated block comment");
    return out;
}

std::vector<std::string> split_operands(const std::string &text) {
    std::vector<std::string> result;
    std::string current;
    int bracket_depth = 0;
    for (char c : text) {
        if (c == '[' || c == '(' || c == '{') ++bracket_depth;
        if (c == ']' || c == ')' || c == '}') --bracket_depth;
        if (c == ',' && bracket_depth == 0) {
            result.push_back(trim(current));
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    if (!trim(current).empty()) result.push_back(trim(current));
    return result;
}

std::string unbracket(std::string s) {
    s = trim(std::move(s));
    if (s.size() >= 2 && s.front() == '[' && s.back() == ']') {
        return trim(s.substr(1, s.size() - 2));
    }
    throw CompileError("expected bracketed address, got '" + s + "'");
}

enum class Type : uint8_t {
    B32 = 0x0, B64 = 0x1, U32 = 0x2, S32 = 0x3, F32 = 0x8, None = 0xf
};

Type parse_type(const std::string &name) {
    if (name == "b32") return Type::B32;
    if (name == "b64") return Type::B64;
    if (name == "u32") return Type::U32;
    if (name == "s32") return Type::S32;
    if (name == "u64") return Type::B64;
    if (name == "f32") return Type::F32;
    if (name == "pred") return Type::None;
    throw CompileError("unsupported PTX type ." + name);
}

bool is_pair_type(Type type) { return type == Type::B64; }

struct Param {
    std::string name;
    Type type;
    uint32_t offset;
};

struct RegisterDecl {
    std::string prefix;
    Type type;
    unsigned count;
};

struct PtxInstruction {
    std::string mnemonic;
    std::vector<std::string> operands;
    std::optional<std::string> guard;
    bool guard_neg = false;
    unsigned line = 0;
};

struct PtxItem {
    std::optional<std::string> label;
    std::optional<PtxInstruction> instruction;
};

struct Program {
    std::string kernel;
    std::vector<Param> params;
    std::vector<RegisterDecl> declarations;
    std::vector<PtxItem> items;
};

uint32_t align_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

unsigned line_number_at(const std::string &text, std::size_t pos) {
    return 1u + static_cast<unsigned>(std::count(text.begin(), text.begin() + pos, '\n'));
}

Program parse_ptx(const std::string &raw) {
    const std::string text = strip_comments(raw);
    Program program;

    std::smatch entry_match;
    std::regex entry_re(R"(\.visible\s+\.entry\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\()",
                        std::regex::ECMAScript);
    if (!std::regex_search(text, entry_match, entry_re)) {
        throw CompileError("missing .visible .entry declaration");
    }
    program.kernel = entry_match[1].str();
    std::size_t open_paren = static_cast<std::size_t>(entry_match.position(0) + entry_match.length(0) - 1);
    int depth = 0;
    std::size_t close_paren = std::string::npos;
    for (std::size_t i = open_paren; i < text.size(); ++i) {
        if (text[i] == '(') ++depth;
        if (text[i] == ')' && --depth == 0) { close_paren = i; break; }
    }
    if (close_paren == std::string::npos) throw CompileError("unterminated kernel parameter list");

    uint32_t param_offset = 0;
    for (const auto &part : split_operands(text.substr(open_paren + 1, close_paren - open_paren - 1))) {
        std::smatch m;
        std::regex param_re(R"(^\.param\s+\.([A-Za-z0-9]+)\s+([A-Za-z_$][A-Za-z0-9_$]*)$)");
        const std::string declaration = trim(part);
        if (!std::regex_match(declaration, m, param_re)) {
            throw CompileError("invalid kernel parameter declaration: " + part);
        }
        Type type = parse_type(m[1].str());
        uint32_t size = is_pair_type(type) ? 8 : 4;
        param_offset = align_up(param_offset, size);
        program.params.push_back({m[2].str(), type, param_offset});
        param_offset += size;
    }

    std::size_t body_open = text.find('{', close_paren);
    if (body_open == std::string::npos) throw CompileError("missing kernel body");
    depth = 0;
    std::size_t body_close = std::string::npos;
    for (std::size_t i = body_open; i < text.size(); ++i) {
        if (text[i] == '{') ++depth;
        if (text[i] == '}' && --depth == 0) { body_close = i; break; }
    }
    if (body_close == std::string::npos) throw CompileError("unterminated kernel body");

    std::string body = text.substr(body_open + 1, body_close - body_open - 1);
    std::size_t cursor = 0;
    while (cursor < body.size()) {
        std::size_t semicolon = body.find(';', cursor);
        std::size_t end = semicolon == std::string::npos ? body.size() : semicolon;
        std::string statement = trim(body.substr(cursor, end - cursor));
        unsigned line = line_number_at(text, body_open + 1 + cursor);
        cursor = semicolon == std::string::npos ? body.size() : semicolon + 1;
        if (statement.empty()) continue;

        while (true) {
            std::smatch label_match;
            std::regex label_re(R"(^([A-Za-z_$][A-Za-z0-9_$.]*)\s*:\s*)");
            if (!std::regex_search(statement, label_match, label_re)) break;
            program.items.push_back({label_match[1].str(), std::nullopt});
            statement = trim(statement.substr(label_match.length(0)));
            if (statement.empty()) break;
        }
        if (statement.empty()) continue;

        if (statement.rfind(".reg", 0) == 0) {
            std::smatch m;
            std::regex reg_re(R"(^\.reg\s+\.([A-Za-z0-9]+)\s+(%[A-Za-z_$][A-Za-z0-9_$]*)(?:<([0-9]+)>)?$)");
            if (!std::regex_match(statement, m, reg_re)) {
                throw CompileError("line " + std::to_string(line) + ": invalid register declaration: " + statement);
            }
            unsigned count = m[3].matched ? static_cast<unsigned>(std::stoul(m[3].str())) : 1;
            program.declarations.push_back({m[2].str(), parse_type(m[1].str()), count});
            continue;
        }
        if (!statement.empty() && statement.front() == '.') {
            throw CompileError("line " + std::to_string(line) + ": unsupported directive in kernel: " + statement);
        }

        PtxInstruction inst;
        inst.line = line;
        if (statement.front() == '@') {
            std::size_t space = statement.find_first_of(" \t\n");
            if (space == std::string::npos) throw CompileError("line " + std::to_string(line) + ": missing guarded instruction");
            std::string guard = statement.substr(1, space - 1);
            if (!guard.empty() && guard.front() == '!') {
                inst.guard_neg = true;
                guard.erase(guard.begin());
            }
            if (guard.empty() || guard.front() != '%') throw CompileError("line " + std::to_string(line) + ": invalid predicate guard");
            inst.guard = guard;
            statement = trim(statement.substr(space));
        }
        std::size_t space = statement.find_first_of(" \t\n");
        inst.mnemonic = space == std::string::npos ? statement : statement.substr(0, space);
        if (space != std::string::npos) inst.operands = split_operands(statement.substr(space));
        program.items.push_back({std::nullopt, std::move(inst)});
    }
    return program;
}

std::optional<std::string> defined_register(const PtxInstruction &inst) {
    if (inst.operands.empty()) return std::nullopt;
    std::string op = inst.mnemonic.substr(0, inst.mnemonic.find('.'));
    if (op == "st" || op == "bra" || op == "ret") return std::nullopt;
    if (!inst.operands[0].empty() && inst.operands[0].front() == '%') return inst.operands[0];
    return std::nullopt;
}

std::vector<std::string> ptx_source_registers(const PtxInstruction &inst);

bool ptx_cse_candidate(const PtxInstruction &inst) {
    if (inst.guard) return false;
    std::string op = inst.mnemonic.substr(0, inst.mnemonic.find('.'));
    static const std::vector<std::string> pure = {
        "add", "sub", "mul", "mad", "fma", "and", "or", "xor", "shl", "shr", "mov"
    };
    if (std::find(pure.begin(), pure.end(), op) != pure.end()) return true;
    return inst.mnemonic.rfind("ld.global.", 0) == 0;
}

Program eliminate_ptx_common_expressions(Program program) {
    std::unordered_map<std::string, std::string> alias;
    std::unordered_map<std::string, std::string> available;
    unsigned memory_epoch = 0;
    auto reset_block = [&]() { alias.clear(); available.clear(); memory_epoch = 0; };
    auto canonical_register = [&](std::string name) {
        for (unsigned depth = 0; depth < 64; ++depth) {
            auto it = alias.find(name);
            if (it == alias.end() || it->second == name) break;
            name = it->second;
        }
        return name;
    };
    auto rewrite_use = [&](std::string operand) {
        std::string value = trim(operand);
        if (!value.empty() && value.front() == '%' && value.find_first_of(" \t") == std::string::npos) {
            return canonical_register(value);
        }
        if (value.size() >= 3 && value.front() == '[' && value.back() == ']') {
            std::string inside = trim(value.substr(1, value.size() - 2));
            if (!inside.empty() && inside.front() == '%') return std::string("[") + canonical_register(inside) + "]";
        }
        return operand;
    };
    auto redefined_later_in_block = [&](const std::string &name, std::size_t from) {
        for (std::size_t j = from + 1; j < program.items.size(); ++j) {
            if (program.items[j].label) break;
            if (!program.items[j].instruction) continue;
            auto definition = defined_register(*program.items[j].instruction);
            if (definition && *definition == name) return true;
            std::string op = program.items[j].instruction->mnemonic.substr(0, program.items[j].instruction->mnemonic.find('.'));
            if (op == "bra" || op == "ret") break;
        }
        return false;
    };
    auto used_beyond_block = [&](const std::string &name, std::size_t from) {
        std::size_t block_end = from + 1;
        while (block_end < program.items.size()) {
            if (program.items[block_end].label) break;
            if (program.items[block_end].instruction) {
                std::string op = program.items[block_end].instruction->mnemonic.substr(
                    0, program.items[block_end].instruction->mnemonic.find('.'));
                if (op == "bra" || op == "ret") { ++block_end; break; }
            }
            ++block_end;
        }
        for (std::size_t j = block_end; j < program.items.size(); ++j) {
            if (!program.items[j].instruction) continue;
            for (const auto &source : ptx_source_registers(*program.items[j].instruction)) {
                if (source == name) return true;
            }
            auto definition = defined_register(*program.items[j].instruction);
            if (definition && *definition == name) return false;
        }
        return false;
    };

    std::vector<PtxItem> output;
    output.reserve(program.items.size());
    for (std::size_t index = 0; index < program.items.size(); ++index) {
        PtxItem item = program.items[index];
        if (item.label) reset_block();
        if (!item.instruction) { output.push_back(std::move(item)); continue; }
        auto &inst = *item.instruction;
        auto definition = defined_register(inst);
        std::size_t first_use = definition ? 1 : 0;
        for (std::size_t operand = first_use; operand < inst.operands.size(); ++operand) {
            inst.operands[operand] = rewrite_use(inst.operands[operand]);
        }

        std::string base_op = inst.mnemonic.substr(0, inst.mnemonic.find('.'));
        if (base_op == "st") {
            ++memory_epoch;
            available.clear(); // Conservative alias handling for all memory expressions.
        }

        bool removed = false;
        if (definition && ptx_cse_candidate(inst)) {
            std::ostringstream key;
            key << inst.mnemonic << ':';
            for (std::size_t operand = 1; operand < inst.operands.size(); ++operand) key << inst.operands[operand] << ',';
            if (inst.mnemonic.rfind("ld.global.", 0) == 0) key << "mem=" << memory_epoch;
            auto found = available.find(key.str());
            if (found != available.end() &&
                !redefined_later_in_block(found->second, index) &&
                !redefined_later_in_block(*definition, index) &&
                !used_beyond_block(*definition, index)) {
                alias[*definition] = found->second;
                removed = true;
            } else {
                for (auto it = alias.begin(); it != alias.end();) {
                    if (it->first == *definition || it->second == *definition) it = alias.erase(it);
                    else ++it;
                }
                for (auto it = available.begin(); it != available.end();) {
                    if (it->second == *definition) it = available.erase(it);
                    else ++it;
                }
                available[key.str()] = *definition;
            }
        } else if (definition) {
            for (auto it = alias.begin(); it != alias.end();) {
                if (it->first == *definition || it->second == *definition) it = alias.erase(it);
                else ++it;
            }
        }
        if (!removed) output.push_back(std::move(item));
        if (base_op == "bra" || base_op == "ret") reset_block();
    }
    program.items = std::move(output);
    return program;
}

Program fuse_ptx_multiply_add(Program program) {
    std::vector<bool> remove(program.items.size(), false);
    auto base_op = [](const PtxInstruction &inst) { return inst.mnemonic.substr(0, inst.mnemonic.find('.')); };
    auto is_f32 = [](const PtxInstruction &inst) {
        return inst.mnemonic.size() >= 4 && inst.mnemonic.substr(inst.mnemonic.size() - 4) == ".f32";
    };
    for (std::size_t i = 0; i < program.items.size(); ++i) {
        if (!program.items[i].instruction || remove[i]) continue;
        const auto multiply = *program.items[i].instruction;
        if (base_op(multiply) != "mul" || !is_f32(multiply) || multiply.guard || multiply.operands.size() != 3) continue;
        const std::string temporary_name = multiply.operands[0];
        const std::string factor_a = multiply.operands[1];
        const std::string factor_b = multiply.operands[2];
        std::optional<std::size_t> use_index;
        bool safe = true;
        for (std::size_t j = i + 1; j < program.items.size(); ++j) {
            if (program.items[j].label) break;
            if (!program.items[j].instruction) continue;
            const auto &candidate = *program.items[j].instruction;
            std::string op = base_op(candidate);
            if (op == "bra" || op == "ret") break;
            auto definition = defined_register(candidate);
            if (definition && (*definition == factor_a || *definition == factor_b || *definition == temporary_name)) {
                safe = false;
                break;
            }
            std::size_t first_use = definition ? 1 : 0;
            for (std::size_t operand = first_use; operand < candidate.operands.size(); ++operand) {
                if (candidate.operands[operand] == temporary_name) {
                    if (use_index) safe = false;
                    else use_index = j;
                }
            }
            if (!safe) break;
        }
        if (!safe || !use_index) continue;
        auto &add = *program.items[*use_index].instruction;
        if (base_op(add) != "add" || !is_f32(add) || add.guard || add.operands.size() != 3) continue;
        std::string accumulator;
        if (add.operands[1] == temporary_name) accumulator = add.operands[2];
        else if (add.operands[2] == temporary_name) accumulator = add.operands[1];
        else continue;
        add.mnemonic = "mad.f32";
        add.operands = {add.operands[0], factor_a, factor_b, accumulator};
        remove[i] = true;
    }
    std::vector<PtxItem> output;
    output.reserve(program.items.size());
    for (std::size_t i = 0; i < program.items.size(); ++i) if (!remove[i]) output.push_back(std::move(program.items[i]));
    program.items = std::move(output);
    return program;
}

std::vector<std::string> ptx_source_registers(const PtxInstruction &inst) {
    std::vector<std::string> result;
    auto definition = defined_register(inst);
    std::size_t first = definition ? 1 : 0;
    std::regex register_re(R"(%[A-Za-z_$][A-Za-z0-9_$.]*)");
    for (std::size_t i = first; i < inst.operands.size(); ++i) {
        for (std::sregex_iterator it(inst.operands[i].begin(), inst.operands[i].end(), register_re), end; it != end; ++it) {
            result.push_back(it->str());
        }
    }
    return result;
}

bool licm_pure(const PtxInstruction &inst) {
    if (inst.guard) return false;
    std::string op = inst.mnemonic.substr(0, inst.mnemonic.find('.'));
    static const std::vector<std::string> pure = {
        "mov", "add", "sub", "mul", "mad", "fma", "and", "or", "xor", "shl", "shr"
    };
    return std::find(pure.begin(), pure.end(), op) != pure.end();
}

Program hoist_loop_invariants(Program program) {
    bool changed = true;
    while (changed) {
        changed = false;
        std::unordered_map<std::string, std::size_t> labels;
        for (std::size_t i = 0; i < program.items.size(); ++i) if (program.items[i].label) labels[*program.items[i].label] = i;
        for (std::size_t branch_index = 0; branch_index < program.items.size() && !changed; ++branch_index) {
            if (!program.items[branch_index].instruction) continue;
            const auto &branch = *program.items[branch_index].instruction;
            if (branch.mnemonic != "bra" || branch.operands.size() != 1) continue;
            auto label = labels.find(branch.operands[0]);
            if (label == labels.end() || label->second >= branch_index) continue;
            std::size_t header = label->second;

            bool internal_label = false;
            for (std::size_t i = header + 1; i < branch_index; ++i) if (program.items[i].label) internal_label = true;
            if (internal_label) continue;
            bool external_entry = false;
            for (std::size_t i = 0; i < header; ++i) {
                if (!program.items[i].instruction) continue;
                const auto &candidate = *program.items[i].instruction;
                if (candidate.mnemonic == "bra" && candidate.operands.size() == 1 && candidate.operands[0] == branch.operands[0]) external_entry = true;
            }
            if (external_entry) continue;

            std::unordered_map<std::string, unsigned> definition_count;
            std::unordered_map<std::string, std::size_t> definition_item;
            bool loop_has_store = false;
            bool loop_has_internal_control = false;
            for (std::size_t i = header + 1; i < branch_index; ++i) {
                if (!program.items[i].instruction) continue;
                const auto &loop_inst = *program.items[i].instruction;
                std::string loop_op = loop_inst.mnemonic.substr(0, loop_inst.mnemonic.find('.'));
                loop_has_store |= loop_op == "st";
                loop_has_internal_control |= loop_op == "bra" || loop_op == "ret";
                auto definition = defined_register(loop_inst);
                if (definition) { ++definition_count[*definition]; definition_item[*definition] = i; }
            }
            std::unordered_map<std::string, bool> used_after;
            for (std::size_t i = branch_index + 1; i < program.items.size(); ++i) {
                if (!program.items[i].instruction) continue;
                for (const auto &source : ptx_source_registers(*program.items[i].instruction)) used_after[source] = true;
            }

            std::vector<std::size_t> invariant_items;
            std::unordered_map<std::string, bool> invariant_values;
            bool added = true;
            while (added) {
                added = false;
                for (std::size_t i = header + 1; i < branch_index; ++i) {
                    if (!program.items[i].instruction ||
                        std::find(invariant_items.begin(), invariant_items.end(), i) != invariant_items.end()) continue;
                    const auto &inst = *program.items[i].instruction;
                    auto definition = defined_register(inst);
                    bool invariant_load = inst.mnemonic.rfind("ld.global.", 0) == 0 &&
                                          !inst.guard && !loop_has_store && !loop_has_internal_control;
                    if (!definition || definition_count[*definition] != 1 || used_after[*definition] ||
                        (!licm_pure(inst) && !invariant_load)) continue;
                    bool sources_invariant = true;
                    for (const auto &source : ptx_source_registers(inst)) {
                        if (definition_count[source] != 0 && !invariant_values[source]) { sources_invariant = false; break; }
                    }
                    if (!sources_invariant) continue;
                    invariant_items.push_back(i);
                    invariant_values[*definition] = true;
                    added = true;
                }
            }
            if (invariant_items.empty()) continue;
            std::sort(invariant_items.begin(), invariant_items.end());
            std::vector<bool> move(program.items.size(), false);
            for (std::size_t index : invariant_items) move[index] = true;
            std::vector<PtxItem> output;
            output.reserve(program.items.size());
            for (std::size_t i = 0; i < program.items.size(); ++i) {
                if (i == header) {
                    for (std::size_t index : invariant_items) output.push_back(std::move(program.items[index]));
                }
                if (!move[i]) output.push_back(std::move(program.items[i]));
            }
            program.items = std::move(output);
            changed = true;
        }
    }
    return program;
}

Program reduce_loop_address_strength(Program program) {
    bool transformed = true;
    unsigned unique_id = 0;
    while (transformed) {
        transformed = false;
        std::unordered_map<std::string, std::size_t> labels;
        for (std::size_t i = 0; i < program.items.size(); ++i) if (program.items[i].label) labels[*program.items[i].label] = i;
        for (std::size_t backedge = 0; backedge < program.items.size() && !transformed; ++backedge) {
            if (!program.items[backedge].instruction) continue;
            const auto &branch = *program.items[backedge].instruction;
            if (branch.mnemonic != "bra" || branch.operands.size() != 1) continue;
            auto label = labels.find(branch.operands[0]);
            if (label == labels.end() || label->second >= backedge) continue;
            std::size_t header = label->second;
            bool internal_label = false;
            for (std::size_t i = header + 1; i < backedge; ++i) if (program.items[i].label) internal_label = true;
            if (internal_label) continue;

            std::unordered_map<std::string, unsigned> definitions;
            std::unordered_map<std::string, bool> induction;
            for (std::size_t i = header + 1; i < backedge; ++i) {
                if (!program.items[i].instruction) continue;
                const auto &inst = *program.items[i].instruction;
                auto definition = defined_register(inst);
                if (definition) ++definitions[*definition];
                if (inst.mnemonic == "add.u32" && inst.operands.size() == 3 &&
                    (inst.operands[0] == inst.operands[1] || inst.operands[0] == inst.operands[2])) {
                    induction[inst.operands[0]] = true;
                }
            }

            struct Rewrite {
                std::size_t first;
                std::string address;
                std::vector<PtxItem> preheader;
                PtxItem increment;
            };
            std::vector<Rewrite> rewrites;
            std::unordered_map<std::string, bool> rewritten_addresses;
            for (std::size_t i = header + 1; i + 2 < backedge; ++i) {
                if (!program.items[i].instruction || !program.items[i + 1].instruction || !program.items[i + 2].instruction) continue;
                const auto &mad = *program.items[i].instruction;
                const auto &wide = *program.items[i + 1].instruction;
                const auto &add64 = *program.items[i + 2].instruction;
                if (mad.mnemonic != "mad.lo.u32" || mad.operands.size() != 4 ||
                    wide.mnemonic != "mul.wide.u32" || wide.operands.size() != 3 ||
                    add64.mnemonic != "add.u64" || add64.operands.size() != 3) continue;
                std::string scale;
                if (wide.operands[1] == mad.operands[0]) scale = wide.operands[2];
                else if (wide.operands[2] == mad.operands[0]) scale = wide.operands[1];
                else continue;
                std::string base;
                if (add64.operands[1] == wide.operands[0]) base = add64.operands[2];
                else if (add64.operands[2] == wide.operands[0]) base = add64.operands[1];
                else continue;
                const std::string &address = add64.operands[0];
                if (rewritten_addresses[address] || definitions[address] != 1) continue;

                std::string iv;
                std::optional<std::string> coefficient;
                if (induction[mad.operands[3]]) iv = mad.operands[3];
                else if (induction[mad.operands[1]]) { iv = mad.operands[1]; coefficient = mad.operands[2]; }
                else if (induction[mad.operands[2]]) { iv = mad.operands[2]; coefficient = mad.operands[1]; }
                else continue;
                (void)iv;
                if (definitions[base] != 0) continue; // Base pointer must be loop invariant.
                if (coefficient && definitions[*coefficient] != 0) continue;

                bool used_after = false;
                for (std::size_t j = backedge + 1; j < program.items.size(); ++j) {
                    if (!program.items[j].instruction) continue;
                    for (const auto &source : ptx_source_registers(*program.items[j].instruction)) {
                        if (source == address) used_after = true;
                    }
                    auto definition = defined_register(*program.items[j].instruction);
                    if (definition && *definition == address) break;
                }
                if (used_after) continue;

                std::string step = "%__aec_step" + std::to_string(unique_id++);
                program.declarations.push_back({step, Type::B64, 1});
                PtxInstruction step_init;
                step_init.line = mad.line;
                if (coefficient) {
                    step_init.mnemonic = "mul.wide.u32";
                    step_init.operands = {step, *coefficient, scale};
                } else {
                    if (!scale.empty() && scale.front() == '%') continue;
                    step_init.mnemonic = "mov.u64";
                    step_init.operands = {step, scale};
                }
                PtxInstruction increment;
                increment.line = add64.line;
                increment.mnemonic = "add.u64";
                increment.operands = {address, address, step};
                rewrites.push_back({i, address,
                    {program.items[i], program.items[i + 1], program.items[i + 2],
                     PtxItem{std::nullopt, std::move(step_init)}},
                    PtxItem{std::nullopt, std::move(increment)}});
                rewritten_addresses[address] = true;
                i += 2;
            }
            if (rewrites.empty()) continue;
            std::vector<bool> remove(program.items.size(), false);
            for (const auto &rewrite : rewrites) {
                remove[rewrite.first] = remove[rewrite.first + 1] = remove[rewrite.first + 2] = true;
            }
            std::vector<PtxItem> output;
            output.reserve(program.items.size() + rewrites.size() * 2);
            for (std::size_t i = 0; i < program.items.size(); ++i) {
                if (i == header) for (auto &rewrite : rewrites) {
                    for (auto &item : rewrite.preheader) output.push_back(std::move(item));
                }
                if (i == backedge) for (auto &rewrite : rewrites) output.push_back(std::move(rewrite.increment));
                if (!remove[i]) output.push_back(std::move(program.items[i]));
            }
            program.items = std::move(output);
            transformed = true;
        }
    }
    return program;
}

Program rotate_guarded_loops(Program program) {
    std::unordered_map<std::string, std::size_t> labels;
    for (std::size_t i = 0; i < program.items.size(); ++i) if (program.items[i].label) labels[*program.items[i].label] = i;
    for (std::size_t backedge = 0; backedge < program.items.size(); ++backedge) {
        if (!program.items[backedge].instruction) continue;
        const auto original_backedge = *program.items[backedge].instruction;
        if (original_backedge.mnemonic != "bra" || original_backedge.guard || original_backedge.operands.size() != 1) continue;
        auto label = labels.find(original_backedge.operands[0]);
        if (label == labels.end() || label->second >= backedge) continue;
        std::size_t header = label->second;

        bool other_entry = false;
        for (std::size_t i = 0; i < program.items.size(); ++i) {
            if (i == backedge || !program.items[i].instruction) continue;
            const auto &candidate = *program.items[i].instruction;
            if (candidate.mnemonic == "bra" && candidate.operands.size() == 1 && candidate.operands[0] == original_backedge.operands[0]) {
                other_entry = true;
            }
        }
        if (other_entry) continue;

        std::size_t compare_index = header + 1;
        while (compare_index < backedge && !program.items[compare_index].instruction) ++compare_index;
        std::size_t exit_index = compare_index + 1;
        while (exit_index < backedge && !program.items[exit_index].instruction) ++exit_index;
        if (exit_index >= backedge) continue;
        const auto compare = *program.items[compare_index].instruction;
        const auto exit_branch = *program.items[exit_index].instruction;
        if (compare.mnemonic.rfind("setp.", 0) != 0 || compare.operands.size() != 3 ||
            exit_branch.mnemonic != "bra" || !exit_branch.guard || exit_branch.guard_neg ||
            exit_branch.operands.size() != 1 || *exit_branch.guard != compare.operands[0]) continue;

        static const std::map<std::string, std::string> inverse = {
            {"eq","ne"},{"ne","eq"},{"lt","ge"},{"le","gt"},{"gt","le"},{"ge","lt"}
        };
        std::size_t first_dot = compare.mnemonic.find('.');
        std::size_t second_dot = compare.mnemonic.find('.', first_dot + 1);
        if (first_dot == std::string::npos || second_dot == std::string::npos) continue;
        std::string relation = compare.mnemonic.substr(first_dot + 1, second_dot - first_dot - 1);
        auto inverted = inverse.find(relation);
        if (inverted == inverse.end()) continue;

        PtxInstruction continue_compare = compare;
        continue_compare.mnemonic = compare.mnemonic.substr(0, first_dot + 1) + inverted->second + compare.mnemonic.substr(second_dot);
        PtxInstruction continue_branch;
        continue_branch.line = original_backedge.line;
        continue_branch.mnemonic = "bra";
        continue_branch.operands = original_backedge.operands;
        continue_branch.guard = compare.operands[0];

        std::vector<PtxItem> output;
        output.reserve(program.items.size() + 1);
        for (std::size_t i = 0; i < program.items.size(); ++i) {
            if (i == header) {
                output.push_back(PtxItem{std::nullopt, compare});
                output.push_back(PtxItem{std::nullopt, exit_branch});
            }
            if (i == compare_index || i == exit_index) continue;
            if (i == backedge) {
                output.push_back(PtxItem{std::nullopt, std::move(continue_compare)});
                output.push_back(PtxItem{std::nullopt, std::move(continue_branch)});
                continue;
            }
            output.push_back(std::move(program.items[i]));
        }
        program.items = std::move(output);
        return program; // Recompute indices before considering another loop.
    }
    return program;
}

Program unroll_rotated_loop_by_four(Program program) {
    std::unordered_map<std::string, std::size_t> labels;
    for (std::size_t i = 0; i < program.items.size(); ++i) if (program.items[i].label) labels[*program.items[i].label] = i;
    unsigned unique = 0;
    for (std::size_t branch_index = 0; branch_index < program.items.size(); ++branch_index) {
        if (!program.items[branch_index].instruction) continue;
        const auto branch = *program.items[branch_index].instruction;
        if (branch.mnemonic != "bra" || !branch.guard || branch.guard_neg || branch.operands.size() != 1) continue;
        auto header_it = labels.find(branch.operands[0]);
        if (header_it == labels.end() || header_it->second >= branch_index) continue;
        std::size_t header = header_it->second;

        std::size_t compare_index = branch_index;
        while (compare_index > header && !program.items[--compare_index].instruction) {}
        if (compare_index <= header || !program.items[compare_index].instruction) continue;
        const auto compare = *program.items[compare_index].instruction;
        if (compare.mnemonic != "setp.lt.u32" || compare.operands.size() != 3 ||
            compare.operands[0] != *branch.guard) continue;
        const std::string iv = compare.operands[1];
        const std::string bound = compare.operands[2];

        std::vector<PtxItem> body;
        bool unsafe = false;
        unsigned iv_updates = 0;
        std::string step;
        for (std::size_t i = header + 1; i < compare_index; ++i) {
            if (program.items[i].label || !program.items[i].instruction) { unsafe = true; break; }
            const auto &inst = *program.items[i].instruction;
            if (inst.mnemonic == "bra" || inst.mnemonic == "ret") { unsafe = true; break; }
            if (inst.mnemonic == "add.u32" && inst.operands.size() == 3 && inst.operands[0] == iv) {
                if (inst.operands[1] == iv) step = inst.operands[2];
                else if (inst.operands[2] == iv) step = inst.operands[1];
                else { unsafe = true; break; }
                ++iv_updates;
            }
            body.push_back(program.items[i]);
        }
        if (unsafe || body.empty() || body.size() > 24 || iv_updates != 1) continue;

        auto has_constant_definition = [&](const std::string &name, const std::string &value) {
            if (name == value) return true;
            bool found = false;
            for (std::size_t i = 0; i < header; ++i) {
                if (!program.items[i].instruction) continue;
                const auto &inst = *program.items[i].instruction;
                auto definition = defined_register(inst);
                if (!definition || *definition != name) continue;
                found = inst.mnemonic == "mov.u32" && inst.operands.size() == 2 && inst.operands[1] == value;
            }
            return found;
        };
        if (!has_constant_definition(iv, "0") || !has_constant_definition(step, "1")) continue;

        bool bound_modified = false;
        for (const auto &item : body) {
            auto definition = defined_register(*item.instruction);
            if (definition && *definition == bound) bound_modified = true;
        }
        if (bound_modified) continue;

        std::string exit_label;
        bool insert_exit_label = true;
        for (std::size_t i = branch_index + 1; i < program.items.size(); ++i) {
            if (program.items[i].label) { exit_label = *program.items[i].label; insert_exit_label = false; break; }
            if (program.items[i].instruction) break;
        }
        std::string suffix = std::to_string(unique++);
        if (exit_label.empty()) exit_label = "__AEC_UNROLL_EXIT_" + suffix;
        std::string tail_check = "__AEC_TAIL_CHECK_" + suffix;
        std::string tail_loop = "__AEC_TAIL_LOOP_" + suffix;
        std::string main_bound = "%__aec_main_bound" + suffix;
        program.declarations.push_back({main_bound, Type::U32, 1});

        auto make_inst = [&](std::string mnemonic, std::vector<std::string> operands,
                             std::optional<std::string> guard = std::nullopt) {
            PtxInstruction inst;
            inst.line = compare.line;
            inst.mnemonic = std::move(mnemonic);
            inst.operands = std::move(operands);
            inst.guard = std::move(guard);
            return PtxItem{std::nullopt, std::move(inst)};
        };
        std::vector<PtxItem> output;
        output.reserve(program.items.size() + body.size() * 4 + 12);
        for (std::size_t i = 0; i < program.items.size(); ++i) {
            if (i == header) {
                output.push_back(make_inst("and.b32", {main_bound, bound, "4294967292"}));
                output.push_back(make_inst("setp.ge.u32", {compare.operands[0], iv, main_bound}));
                output.push_back(make_inst("bra", {tail_check}, compare.operands[0]));
                output.push_back(std::move(program.items[i]));
                for (unsigned copy = 0; copy < 4; ++copy) for (const auto &item : body) output.push_back(item);
                i = compare_index - 1;
                continue;
            }
            if (i == compare_index) {
                auto main_compare = compare;
                main_compare.operands[2] = main_bound;
                output.push_back(PtxItem{std::nullopt, std::move(main_compare)});
                output.push_back(std::move(program.items[branch_index]));
                output.push_back(PtxItem{tail_check, std::nullopt});
                output.push_back(make_inst("setp.ge.u32", {compare.operands[0], iv, bound}));
                output.push_back(make_inst("bra", {exit_label}, compare.operands[0]));
                output.push_back(PtxItem{tail_loop, std::nullopt});
                for (const auto &item : body) output.push_back(item);
                output.push_back(PtxItem{std::nullopt, compare});
                output.push_back(make_inst("bra", {tail_loop}, compare.operands[0]));
                i = branch_index;
                if (insert_exit_label) output.push_back(PtxItem{exit_label, std::nullopt});
                continue;
            }
            output.push_back(std::move(program.items[i]));
        }
        program.items = std::move(output);
        return program;
    }
    return program;
}

enum Opcode : uint16_t {
    ADD = 0x0001, SUB = 0x0002, MUL = 0x0003, MAD = 0x0004, FMA = 0x0005,
    AND = 0x0010, OR = 0x0011, XOR = 0x0012, SHL = 0x0014, SHR = 0x0015,
    CMPP = 0x0021, LD = 0x0030, ST = 0x0031, BR = 0x0040, BRX = 0x0041,
    HALT = 0x0045, CPY = 0x0054, LOADI = 0x0055, LOADI64 = 0x0056
};

struct MachineInstruction {
    uint16_t opcode = 0;
    uint16_t ctrl = 0;
    uint16_t dest = 0;
    uint16_t src1 = 0;
    uint32_t src2 = 0;
    uint32_t imm = 0;
    std::optional<std::string> branch_label;
};

struct RegisterValue {
    Type type;
    uint16_t physical;
    bool spilled = false;
    uint32_t spill_offset = 0;
};

struct LiveInterval {
    std::string name;
    Type type = Type::None;
    unsigned begin = 0;
    unsigned end = 0;
    unsigned width = 1;
    unsigned references = 0;
    uint16_t physical = 0;
    bool predicate = false;
};

struct CompileStats {
    unsigned ptx_instructions = 0;
    unsigned basic_blocks = 1;
    unsigned virtual_registers = 0;
    unsigned physical_registers = 0;
    unsigned predicates = 0;
    unsigned spill_loads = 0;
    unsigned spill_stores = 0;
};

struct RegisterEffects {
    std::bitset<256> uses;
    std::bitset<256> defs;
    std::bitset<8> pred_uses;
    std::bitset<8> pred_defs;
    bool removable = false;
};

Type machine_type(const MachineInstruction &inst) {
    return static_cast<Type>((inst.ctrl >> 3) & 0xfu);
}

void add_pair_if_needed(std::bitset<256> &set, uint16_t base, Type type) {
    if (base < 256) set.set(base);
    if (is_pair_type(type) && base + 1 < 256) set.set(base + 1);
}

RegisterEffects effects(const MachineInstruction &inst) {
    RegisterEffects result;
    Type type = machine_type(inst);
    if (inst.ctrl & (1u << 15)) result.pred_uses.set(inst.ctrl & 7u);
    auto use = [&](uint16_t reg) { if (reg < 256) result.uses.set(reg); };
    auto def = [&](uint16_t reg) { if (reg < 256) result.defs.set(reg); };
    switch (inst.opcode) {
    case Opcode::ADD: case Opcode::SUB: case Opcode::MUL:
    case Opcode::AND: case Opcode::OR: case Opcode::XOR:
    case Opcode::SHL: case Opcode::SHR:
        use(inst.src1); use(static_cast<uint16_t>(inst.src2)); def(inst.dest); result.removable = true; break;
    case Opcode::MAD: case Opcode::FMA:
        use(inst.src1); use(static_cast<uint16_t>(inst.src2)); use(static_cast<uint16_t>(inst.imm));
        def(inst.dest); result.removable = true; break;
    case Opcode::CMPP:
        use(inst.src1); use(static_cast<uint16_t>(inst.src2));
        if (inst.dest < 8) result.pred_defs.set(inst.dest);
        result.removable = true;
        break;
    case Opcode::LD:
        use(inst.src1); add_pair_if_needed(result.defs, inst.dest, type); result.removable = true; break;
    case Opcode::ST:
        use(inst.src1); use(static_cast<uint16_t>(inst.src2)); break;
    case Opcode::CPY:
        if (inst.src1 < 256) add_pair_if_needed(result.uses, inst.src1, type);
        add_pair_if_needed(result.defs, inst.dest, type); result.removable = true; break;
    case Opcode::LOADI:
        def(inst.dest); result.removable = true; break;
    case Opcode::LOADI64:
        def(inst.dest); if (inst.dest + 1 < 256) def(inst.dest + 1); result.removable = true; break;
    default: break;
    }
    return result;
}

std::vector<MachineInstruction> compact_code(const std::vector<MachineInstruction> &code,
                                             const std::vector<bool> &keep) {
    std::vector<uint32_t> prefix(code.size() + 1, 0);
    for (std::size_t i = 0; i < code.size(); ++i) prefix[i + 1] = prefix[i] + (keep[i] ? 1u : 0u);
    std::vector<MachineInstruction> result;
    result.reserve(prefix.back());
    for (std::size_t i = 0; i < code.size(); ++i) {
        if (!keep[i]) continue;
        MachineInstruction inst = code[i];
        if (inst.opcode == Opcode::BR || inst.opcode == Opcode::BRX) {
            if (inst.imm >= code.size()) throw CompileError("branch target outside program during optimization");
            inst.imm = prefix[inst.imm];
        }
        result.push_back(inst);
    }
    return result;
}

std::vector<std::vector<std::size_t>> successors(const std::vector<MachineInstruction> &code) {
    std::vector<std::vector<std::size_t>> result(code.size());
    for (std::size_t i = 0; i < code.size(); ++i) {
        const auto &inst = code[i];
        if (inst.opcode == Opcode::BR || inst.opcode == Opcode::BRX) {
            if (inst.imm >= code.size()) throw CompileError("branch target outside program");
            result[i].push_back(inst.imm);
        }
        if (inst.opcode != Opcode::BR && inst.opcode != Opcode::HALT && i + 1 < code.size()) {
            result[i].push_back(i + 1);
        }
    }
    return result;
}

unsigned count_basic_blocks(const std::vector<MachineInstruction> &code) {
    if (code.empty()) return 0;
    std::vector<bool> starts(code.size(), false);
    starts[0] = true;
    for (std::size_t i = 0; i < code.size(); ++i) {
        if ((code[i].opcode == Opcode::BR || code[i].opcode == Opcode::BRX) &&
            code[i].imm < code.size()) starts[code[i].imm] = true;
        if ((code[i].opcode == Opcode::BR || code[i].opcode == Opcode::BRX ||
             code[i].opcode == Opcode::HALT) && i + 1 < code.size()) starts[i + 1] = true;
    }
    return static_cast<unsigned>(std::count(starts.begin(), starts.end(), true));
}

std::vector<MachineInstruction> eliminate_dead_code(std::vector<MachineInstruction> code) {
    bool changed = true;
    while (changed) {
        changed = false;
        auto succ = successors(code);
        std::vector<std::bitset<256>> live_in(code.size()), live_out(code.size());
        std::vector<std::bitset<8>> pred_in(code.size()), pred_out(code.size());
        bool dataflow_changed = true;
        while (dataflow_changed) {
            dataflow_changed = false;
            for (std::size_t n = code.size(); n-- > 0;) {
                std::bitset<256> out;
                std::bitset<8> pout;
                for (std::size_t s : succ[n]) { out |= live_in[s]; pout |= pred_in[s]; }
                RegisterEffects effect = effects(code[n]);
                auto in = effect.uses | (out & ~effect.defs);
                auto pin = effect.pred_uses | (pout & ~effect.pred_defs);
                if (out != live_out[n] || in != live_in[n] || pout != pred_out[n] || pin != pred_in[n]) {
                    live_out[n] = out; live_in[n] = in; pred_out[n] = pout; pred_in[n] = pin;
                    dataflow_changed = true;
                }
            }
        }
        std::vector<bool> keep(code.size(), true);
        for (std::size_t i = 0; i < code.size(); ++i) {
            RegisterEffects effect = effects(code[i]);
            if (!effect.removable) continue;
            bool gpr_dead = (effect.defs & live_out[i]).none();
            bool pred_dead = (effect.pred_defs & pred_out[i]).none();
            if (gpr_dead && pred_dead) { keep[i] = false; changed = true; }
        }
        if (changed) code = compact_code(code, keep);
    }
    return code;
}

bool cse_candidate(const MachineInstruction &inst) {
    switch (inst.opcode) {
    case Opcode::ADD: case Opcode::SUB: case Opcode::MUL: case Opcode::MAD: case Opcode::FMA:
    case Opcode::AND: case Opcode::OR: case Opcode::XOR: case Opcode::SHL: case Opcode::SHR:
    case Opcode::CPY: case Opcode::LOADI: case Opcode::LOADI64: case Opcode::LD:
        return (inst.ctrl & (1u << 15)) == 0;
    default: return false;
    }
}

std::string expression_key(const MachineInstruction &inst, const std::vector<unsigned> &versions,
                           unsigned memory_epoch) {
    std::ostringstream key;
    key << inst.opcode << ':' << inst.ctrl << ':';
    auto versioned = [&](uint16_t reg) {
        key << reg << '@';
        // CPY also encodes immutable special-register selectors in src1;
        // only architectural GPR operands have SSA-like version numbers.
        if (reg < versions.size()) key << versions[reg];
        else key << "special";
        key << ':';
    };
    switch (inst.opcode) {
    case Opcode::LOADI: key << inst.imm; break;
    case Opcode::LOADI64: key << inst.src2 << ':' << inst.imm; break;
    case Opcode::CPY: versioned(inst.src1); break;
    case Opcode::LD: versioned(inst.src1); key << "m" << memory_epoch; break;
    case Opcode::MAD: case Opcode::FMA:
        versioned(inst.src1); versioned(static_cast<uint16_t>(inst.src2)); versioned(static_cast<uint16_t>(inst.imm)); break;
    default:
        versioned(inst.src1); versioned(static_cast<uint16_t>(inst.src2)); break;
    }
    return key.str();
}

std::vector<MachineInstruction> eliminate_common_expressions(std::vector<MachineInstruction> code) {
    std::vector<bool> starts(code.size(), false);
    if (!code.empty()) starts[0] = true;
    for (std::size_t i = 0; i < code.size(); ++i) {
        if ((code[i].opcode == Opcode::BR || code[i].opcode == Opcode::BRX) && code[i].imm < code.size()) starts[code[i].imm] = true;
        if ((code[i].opcode == Opcode::BR || code[i].opcode == Opcode::BRX || code[i].opcode == Opcode::HALT) && i + 1 < code.size()) starts[i + 1] = true;
    }
    std::vector<bool> keep(code.size(), true);
    std::size_t begin = 0;
    while (begin < code.size()) {
        std::size_t end = begin + 1;
        while (end < code.size() && !starts[end]) ++end;
        std::unordered_map<std::string, uint16_t> available;
        std::unordered_map<uint16_t, uint16_t> alias;
        std::vector<unsigned> versions(256, 0);
        unsigned memory_epoch = 0;
        auto canonical = [&](uint16_t reg) {
            auto it = alias.find(reg);
            return it == alias.end() ? reg : it->second;
        };
        auto used_beyond_block = [&](uint16_t reg) {
            for (std::size_t j = end; j < code.size(); ++j) {
                RegisterEffects future = effects(code[j]);
                if (future.uses.test(reg)) return true;
                if (future.defs.test(reg)) return false;
            }
            return false;
        };
        auto invalidate_definition = [&](uint16_t reg) {
            alias.erase(reg);
            for (auto it = alias.begin(); it != alias.end();) {
                if (it->second == reg) it = alias.erase(it);
                else ++it;
            }
            for (auto it = available.begin(); it != available.end();) {
                if (it->second == reg) it = available.erase(it);
                else ++it;
            }
            ++versions[reg];
        };
        for (std::size_t i = begin; i < end; ++i) {
            auto &inst = code[i];
            if (inst.opcode == Opcode::ST) {
                inst.src1 = canonical(inst.src1);
                inst.src2 = canonical(static_cast<uint16_t>(inst.src2));
                ++memory_epoch;
                continue;
            }
            if (inst.opcode == Opcode::CMPP) {
                inst.src1 = canonical(inst.src1);
                inst.src2 = canonical(static_cast<uint16_t>(inst.src2));
            } else if (inst.opcode == Opcode::ADD || inst.opcode == Opcode::SUB || inst.opcode == Opcode::MUL ||
                       inst.opcode == Opcode::AND || inst.opcode == Opcode::OR || inst.opcode == Opcode::XOR ||
                       inst.opcode == Opcode::SHL || inst.opcode == Opcode::SHR) {
                inst.src1 = canonical(inst.src1);
                inst.src2 = canonical(static_cast<uint16_t>(inst.src2));
            } else if (inst.opcode == Opcode::MAD || inst.opcode == Opcode::FMA) {
                inst.src1 = canonical(inst.src1);
                inst.src2 = canonical(static_cast<uint16_t>(inst.src2));
                inst.imm = canonical(static_cast<uint16_t>(inst.imm));
            } else if (inst.opcode == Opcode::LD || inst.opcode == Opcode::CPY) {
                if (inst.src1 < 256) inst.src1 = canonical(inst.src1);
            }

            RegisterEffects effect = effects(inst);
            if (!cse_candidate(inst) || effect.defs.none()) {
                for (std::size_t d = 0; d < 256; ++d) if (effect.defs.test(d)) {
                    invalidate_definition(static_cast<uint16_t>(d));
                }
                continue;
            }
            std::string key = expression_key(inst, versions, memory_epoch);
            auto found = available.find(key);
            bool prior_stable = false;
            if (found != available.end()) {
                prior_stable = true;
                for (std::size_t j = i + 1; j < end; ++j) {
                    RegisterEffects future = effects(code[j]);
                    if (future.defs.test(inst.dest)) break; // The aliased value is dead after its next definition.
                    if (future.defs.test(found->second)) { prior_stable = false; break; }
                }
            }
            if (found != available.end() && prior_stable && effect.defs.count() == 1 &&
                !used_beyond_block(inst.dest)) {
                keep[i] = false;
                alias[inst.dest] = found->second;
            } else {
                for (std::size_t d = 0; d < 256; ++d) if (effect.defs.test(d)) {
                    invalidate_definition(static_cast<uint16_t>(d));
                }
                available[key] = inst.dest;
            }
        }
        begin = end;
    }
    return compact_code(code, keep);
}

std::vector<MachineInstruction> fuse_multiply_add(std::vector<MachineInstruction> code) {
    if (code.size() < 2) return code;
    std::vector<unsigned> use_count(256, 0);
    std::vector<bool> block_start(code.size(), false);
    block_start[0] = true;
    for (const auto &inst : code) {
        RegisterEffects effect = effects(inst);
        for (unsigned reg = 0; reg < 256; ++reg) if (effect.uses.test(reg)) ++use_count[reg];
    }
    for (std::size_t i = 0; i < code.size(); ++i) {
        if ((code[i].opcode == Opcode::BR || code[i].opcode == Opcode::BRX) && code[i].imm < code.size()) {
            block_start[code[i].imm] = true;
        }
        if ((code[i].opcode == Opcode::BR || code[i].opcode == Opcode::BRX || code[i].opcode == Opcode::HALT) && i + 1 < code.size()) {
            block_start[i + 1] = true;
        }
    }
    std::vector<bool> keep(code.size(), true);
    for (std::size_t i = 0; i + 1 < code.size(); ++i) {
        MachineInstruction &multiply = code[i];
        MachineInstruction &add = code[i + 1];
        if (block_start[i + 1]) continue;
        if (multiply.opcode != Opcode::MUL || add.opcode != Opcode::ADD) continue;
        if (machine_type(multiply) != Type::F32 || machine_type(add) != Type::F32) continue;
        if ((multiply.ctrl & (1u << 15)) || (add.ctrl & (1u << 15))) continue;
        if (multiply.dest >= 256 || use_count[multiply.dest] != 1) continue;
        uint16_t add_left = add.src1;
        uint16_t add_right = static_cast<uint16_t>(add.src2);
        uint16_t accumulator = 0;
        if (add_left == multiply.dest) accumulator = add_right;
        else if (add_right == multiply.dest) accumulator = add_left;
        else continue;
        add.opcode = Opcode::MAD;
        add.src1 = multiply.src1;
        add.src2 = multiply.src2;
        add.imm = accumulator;
        keep[i] = false;
        ++i;
    }
    return compact_code(code, keep);
}

bool is_memory_instruction(const MachineInstruction &inst) {
    return inst.opcode == Opcode::LD || inst.opcode == Opcode::ST;
}

unsigned instruction_latency(const MachineInstruction &inst) {
    if (inst.opcode == Opcode::LD) return ((inst.ctrl >> 11) & 7u) == 0 ? 20u : 4u;
    if (inst.opcode == Opcode::MUL || inst.opcode == Opcode::MAD || inst.opcode == Opcode::FMA) return 4;
    return 1;
}

std::vector<MachineInstruction> schedule_instructions(std::vector<MachineInstruction> code) {
    if (code.size() < 2) return code;
    std::vector<bool> starts(code.size(), false);
    starts[0] = true;
    for (std::size_t i = 0; i < code.size(); ++i) {
        if ((code[i].opcode == Opcode::BR || code[i].opcode == Opcode::BRX) && code[i].imm < code.size()) starts[code[i].imm] = true;
        if ((code[i].opcode == Opcode::BR || code[i].opcode == Opcode::BRX || code[i].opcode == Opcode::HALT) && i + 1 < code.size()) starts[i + 1] = true;
    }
    std::size_t begin = 0;
    while (begin < code.size()) {
        std::size_t end = begin + 1;
        while (end < code.size() && !starts[end]) ++end;
        std::size_t sched_end = end;
        if (sched_end > begin && (code[sched_end - 1].opcode == Opcode::BR ||
                                  code[sched_end - 1].opcode == Opcode::BRX ||
                                  code[sched_end - 1].opcode == Opcode::HALT)) --sched_end;
        std::size_t count = sched_end - begin;
        // The exact dependency graph is quadratic in basic-block length.
        // Large generated straight-line blocks retain their already-correct
        // source order instead of risking the evaluator's 512 MiB limit.
        constexpr std::size_t max_scheduled_block_instructions = 512;
        if (count > max_scheduled_block_instructions) {
            begin = end;
            continue;
        }
        if (count > 1) {
            std::vector<std::vector<std::size_t>> outgoing(count);
            std::vector<unsigned> indegree(count, 0);
            std::vector<RegisterEffects> block_effects;
            block_effects.reserve(count);
            for (std::size_t i = 0; i < count; ++i) block_effects.push_back(effects(code[begin + i]));
            for (std::size_t i = 0; i < count; ++i) {
                for (std::size_t j = i + 1; j < count; ++j) {
                    const auto &a = block_effects[i];
                    const auto &b = block_effects[j];
                    bool dependent = (a.defs & b.uses).any() || (a.uses & b.defs).any() || (a.defs & b.defs).any() ||
                                     (a.pred_defs & b.pred_uses).any() || (a.pred_uses & b.pred_defs).any() ||
                                     (a.pred_defs & b.pred_defs).any();
                    if (is_memory_instruction(code[begin + i]) && is_memory_instruction(code[begin + j]) &&
                        (code[begin + i].opcode == Opcode::ST || code[begin + j].opcode == Opcode::ST)) dependent = true;
                    if (dependent) { outgoing[i].push_back(j); ++indegree[j]; }
                }
            }
            std::vector<unsigned> height(count, 0);
            for (std::size_t i = count; i-- > 0;) {
                unsigned successor_height = 0;
                for (std::size_t next : outgoing[i]) successor_height = std::max(successor_height, height[next]);
                height[i] = instruction_latency(code[begin + i]) + successor_height;
            }
            std::vector<MachineInstruction> scheduled;
            scheduled.reserve(count);
            std::vector<bool> emitted(count, false);
            for (std::size_t slot = 0; slot < count; ++slot) {
                std::optional<std::size_t> best;
                for (std::size_t i = 0; i < count; ++i) {
                    if (emitted[i] || indegree[i] != 0) continue;
                    if (!best || height[i] > height[*best] ||
                        (height[i] == height[*best] && code[begin + i].opcode == Opcode::LD && code[begin + *best].opcode != Opcode::LD) ||
                        (height[i] == height[*best] && code[begin + i].opcode == code[begin + *best].opcode && i < *best)) best = i;
                }
                if (!best) throw CompileError("cyclic dependency graph during scheduling");
                emitted[*best] = true;
                scheduled.push_back(code[begin + *best]);
                for (std::size_t next : outgoing[*best]) --indegree[next];
            }
            std::copy(scheduled.begin(), scheduled.end(), code.begin() + static_cast<std::ptrdiff_t>(begin));
        }
        begin = end;
    }
    return code;
}

std::vector<MachineInstruction> fold_constants_and_branches(std::vector<MachineInstruction> code) {
    if (code.empty()) return code;
    std::vector<bool> starts(code.size(), false), keep(code.size(), true);
    starts[0] = true;
    for (std::size_t i = 0; i < code.size(); ++i) {
        if ((code[i].opcode == Opcode::BR || code[i].opcode == Opcode::BRX) && code[i].imm < code.size()) starts[code[i].imm] = true;
        if ((code[i].opcode == Opcode::BR || code[i].opcode == Opcode::BRX || code[i].opcode == Opcode::HALT) && i + 1 < code.size()) starts[i + 1] = true;
    }
    std::unordered_map<uint16_t, uint32_t> constants;
    std::unordered_map<uint8_t, bool> predicate_constants;
    auto invalidate_defs = [&](const MachineInstruction &inst) {
        RegisterEffects effect = effects(inst);
        for (unsigned reg = 0; reg < 256; ++reg) if (effect.defs.test(reg)) constants.erase(static_cast<uint16_t>(reg));
        for (unsigned pred = 0; pred < 8; ++pred) if (effect.pred_defs.test(pred)) predicate_constants.erase(static_cast<uint8_t>(pred));
    };
    for (std::size_t i = 0; i < code.size(); ++i) {
        if (starts[i]) { constants.clear(); predicate_constants.clear(); }
        MachineInstruction &inst = code[i];
        if (inst.ctrl & (1u << 15)) {
            uint8_t pred = inst.ctrl & 7u;
            auto known = predicate_constants.find(pred);
            if (known != predicate_constants.end()) {
                bool executes = known->second ^ bool(inst.ctrl & (1u << 14));
                if (!executes) { keep[i] = false; continue; }
                inst.ctrl &= static_cast<uint16_t>(~((1u << 15) | (1u << 14) | 7u));
                if (inst.opcode == Opcode::BRX) inst.opcode = Opcode::BR;
            }
        }
        invalidate_defs(inst);
        Type type = machine_type(inst);
        if (inst.opcode == Opcode::LOADI) {
            constants[inst.dest] = inst.imm;
            continue;
        }
        if (inst.opcode == Opcode::LOADI64) {
            constants[inst.dest] = inst.imm;
            if (inst.dest + 1 < 256) constants[inst.dest + 1] = inst.src2;
            continue;
        }
        if (inst.opcode == Opcode::CPY && inst.src1 < 256) {
            auto value = constants.find(inst.src1);
            auto high = constants.find(static_cast<uint16_t>(inst.src1 + 1));
            if (type == Type::B64 && inst.src1 + 1 < 256 && value != constants.end() && high != constants.end()) {
                inst.opcode = Opcode::LOADI64;
                inst.ctrl = static_cast<uint16_t>(static_cast<unsigned>(Type::None) << 3);
                inst.src1 = 0; inst.src2 = high->second; inst.imm = value->second;
                constants[inst.dest] = value->second;
                if (inst.dest + 1 < 256) constants[inst.dest + 1] = high->second;
            } else if (type != Type::B64 && value != constants.end()) {
                inst.opcode = Opcode::LOADI;
                inst.ctrl = static_cast<uint16_t>(static_cast<unsigned>(Type::None) << 3);
                inst.src1 = 0; inst.src2 = 0; inst.imm = value->second;
                constants[inst.dest] = value->second;
            }
            continue;
        }
        auto a = constants.find(inst.src1);
        auto b = constants.find(static_cast<uint16_t>(inst.src2));
        bool integer_type = type == Type::U32 || type == Type::S32 || type == Type::B32;
        std::optional<uint32_t> result;
        if (integer_type && a != constants.end() && b != constants.end()) {
            switch (inst.opcode) {
            case Opcode::ADD: result = a->second + b->second; break;
            case Opcode::SUB: result = a->second - b->second; break;
            case Opcode::MUL: result = static_cast<uint32_t>(uint64_t(a->second) * b->second); break;
            case Opcode::AND: result = a->second & b->second; break;
            case Opcode::OR: result = a->second | b->second; break;
            case Opcode::XOR: result = a->second ^ b->second; break;
            case Opcode::SHL: result = a->second << (b->second & 31); break;
            case Opcode::SHR:
                if (type == Type::S32) result = static_cast<uint32_t>(static_cast<int32_t>(a->second) >> (b->second & 31));
                else result = a->second >> (b->second & 31);
                break;
            case Opcode::MAD: {
                auto c = constants.find(static_cast<uint16_t>(inst.imm));
                if (c != constants.end()) result = static_cast<uint32_t>(uint64_t(a->second) * b->second + c->second);
                break;
            }
            default: break;
            }
        }
        if (result) {
            inst.opcode = Opcode::LOADI;
            inst.ctrl = static_cast<uint16_t>(static_cast<unsigned>(Type::None) << 3);
            inst.src1 = 0; inst.src2 = 0; inst.imm = *result;
            constants[inst.dest] = *result;
            continue;
        }
        if (integer_type && (inst.opcode == Opcode::ADD || inst.opcode == Opcode::SUB ||
                             inst.opcode == Opcode::MUL || inst.opcode == Opcode::AND ||
                             inst.opcode == Opcode::OR || inst.opcode == Opcode::XOR ||
                             inst.opcode == Opcode::SHL || inst.opcode == Opcode::SHR)) {
            std::optional<uint16_t> copy_source;
            std::optional<uint32_t> immediate;
            bool same_source = inst.src1 == static_cast<uint16_t>(inst.src2);
            if (inst.opcode == Opcode::ADD) {
                if (a != constants.end() && a->second == 0) copy_source = static_cast<uint16_t>(inst.src2);
                else if (b != constants.end() && b->second == 0) copy_source = inst.src1;
            } else if (inst.opcode == Opcode::SUB) {
                if (same_source) immediate = 0;
                else if (b != constants.end() && b->second == 0) copy_source = inst.src1;
            } else if (inst.opcode == Opcode::MUL) {
                if ((a != constants.end() && a->second == 0) || (b != constants.end() && b->second == 0)) immediate = 0;
                else if (a != constants.end() && a->second == 1) copy_source = static_cast<uint16_t>(inst.src2);
                else if (b != constants.end() && b->second == 1) copy_source = inst.src1;
            } else if (inst.opcode == Opcode::AND) {
                if ((a != constants.end() && a->second == 0) || (b != constants.end() && b->second == 0)) immediate = 0;
                else if (a != constants.end() && a->second == 0xffffffffu) copy_source = static_cast<uint16_t>(inst.src2);
                else if (b != constants.end() && b->second == 0xffffffffu) copy_source = inst.src1;
                else if (same_source) copy_source = inst.src1;
            } else if (inst.opcode == Opcode::OR) {
                if (a != constants.end() && a->second == 0) copy_source = static_cast<uint16_t>(inst.src2);
                else if (b != constants.end() && b->second == 0) copy_source = inst.src1;
                else if (same_source) copy_source = inst.src1;
            } else if (inst.opcode == Opcode::XOR) {
                if (same_source) immediate = 0;
                else if (a != constants.end() && a->second == 0) copy_source = static_cast<uint16_t>(inst.src2);
                else if (b != constants.end() && b->second == 0) copy_source = inst.src1;
            } else if ((inst.opcode == Opcode::SHL || inst.opcode == Opcode::SHR) &&
                       b != constants.end() && (b->second & 31u) == 0) {
                copy_source = inst.src1;
            }
            if (immediate) {
                inst.opcode = Opcode::LOADI;
                inst.ctrl = static_cast<uint16_t>(static_cast<unsigned>(Type::None) << 3);
                inst.src1 = 0; inst.src2 = 0; inst.imm = *immediate;
                constants[inst.dest] = *immediate;
                continue;
            }
            if (copy_source) {
                inst.opcode = Opcode::CPY;
                inst.src1 = *copy_source; inst.src2 = 0; inst.imm = 0;
                auto copied = constants.find(*copy_source);
                if (copied != constants.end()) constants[inst.dest] = copied->second;
                continue;
            }
        }
        if (inst.opcode == Opcode::CMPP && a != constants.end() && b != constants.end()) {
            unsigned relation = (inst.ctrl >> 8) & 7u;
            bool value = false;
            if (type == Type::S32) {
                int32_t lhs = static_cast<int32_t>(a->second), rhs = static_cast<int32_t>(b->second);
                if (relation == 0) value = lhs == rhs; else if (relation == 1) value = lhs != rhs;
                else if (relation == 2) value = lhs < rhs; else if (relation == 3) value = lhs <= rhs;
                else if (relation == 4) value = lhs > rhs; else if (relation == 5) value = lhs >= rhs;
            } else {
                uint32_t lhs = a->second, rhs = b->second;
                if (relation == 0) value = lhs == rhs; else if (relation == 1) value = lhs != rhs;
                else if (relation == 2) value = lhs < rhs; else if (relation == 3) value = lhs <= rhs;
                else if (relation == 4) value = lhs > rhs; else if (relation == 5) value = lhs >= rhs;
            }
            predicate_constants[static_cast<uint8_t>(inst.dest)] = value;
        }
    }
    return compact_code(code, keep);
}

std::vector<MachineInstruction> eliminate_unreachable_code(std::vector<MachineInstruction> code) {
    if (code.empty()) return code;
    auto succ = successors(code);
    std::vector<bool> reachable(code.size(), false);
    std::vector<std::size_t> worklist = {0};
    reachable[0] = true;
    while (!worklist.empty()) {
        std::size_t node = worklist.back();
        worklist.pop_back();
        for (std::size_t next : succ[node]) {
            if (!reachable[next]) { reachable[next] = true; worklist.push_back(next); }
        }
    }
    return compact_code(code, reachable);
}

std::vector<MachineInstruction> eliminate_redundant_branches(std::vector<MachineInstruction> code) {
    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<bool> keep(code.size(), true);
        for (std::size_t i = 0; i < code.size(); ++i) {
            if ((code[i].opcode == Opcode::BR || code[i].opcode == Opcode::BRX) &&
                code[i].imm == i + 1) {
                keep[i] = false;
                changed = true;
            }
        }
        if (changed) code = compact_code(code, keep);
    }
    return code;
}

class Lowerer {
public:
    explicit Lowerer(const Program &program, bool pool_constants) : program_(program), pool_constants_(pool_constants) {
        for (const auto &decl : program.declarations) declarations_.push_back(decl);
        for (const auto &param : program.params) params_[param.name] = param;
    }

    std::pair<std::vector<MachineInstruction>, CompileStats> run() {
        plan_register_layout();
        current_instruction_ = 0;
        for (const auto &[value, physical] : constant_registers_) {
            emit(Opcode::LOADI, Type::None, physical, 0, 0, value);
        }
        for (const auto &item : program_.items) {
            if (item.label) {
                if (labels_.count(*item.label)) fail(0, "duplicate label '" + *item.label + "'");
                labels_[*item.label] = code_.size();
                ++label_count_;
            }
            if (item.instruction) {
                ++stats_.ptx_instructions;
                scratch_registers_.clear();
                lower(*item.instruction);
                flush_spill_writes(item.instruction->line);
                ++current_instruction_;
            }
        }
        for (auto &inst : code_) {
            if (!inst.branch_label) continue;
            auto it = labels_.find(*inst.branch_label);
            if (it == labels_.end()) fail(0, "undefined branch label '" + *inst.branch_label + "'");
            inst.imm = static_cast<uint32_t>(it->second);
        }
        stats_.basic_blocks = std::max(1u, label_count_ + 1);
        stats_.virtual_registers = static_cast<unsigned>(registers_.size() + predicates_.size());
        stats_.physical_registers = physical_high_water_;
        stats_.predicates = predicate_high_water_;
        return {code_, stats_};
    }

private:
    const Program &program_;
    bool pool_constants_ = false;
    std::vector<RegisterDecl> declarations_;
    std::unordered_map<std::string, Param> params_;
    std::unordered_map<std::string, RegisterValue> registers_;
    std::unordered_map<std::string, uint8_t> predicates_;
    std::unordered_map<std::string, std::size_t> labels_;
    std::vector<MachineInstruction> code_;
    CompileStats stats_;
    std::vector<LiveInterval> intervals_;
    std::vector<uint16_t> scratch_registers_;
    std::map<uint32_t, uint16_t> constant_registers_;
    std::vector<std::pair<std::string, uint16_t>> pending_spill_writes_;
    unsigned current_instruction_ = 0;
    uint16_t physical_high_water_ = 0;
    uint8_t predicate_high_water_ = 0;
    std::optional<uint8_t> active_guard_;
    bool active_guard_neg_ = false;
    unsigned label_count_ = 0;

    [[noreturn]] void fail(unsigned line, const std::string &message) const {
        if (line) throw CompileError("line " + std::to_string(line) + ": " + message);
        throw CompileError(message);
    }

    RegisterDecl declaration_for(const std::string &name, unsigned line) const {
        RegisterDecl best;
        bool found = false;
        for (const auto &decl : declarations_) {
            if (name.rfind(decl.prefix, 0) != 0) continue;
            std::string suffix = name.substr(decl.prefix.size());
            if (decl.count == 1 && suffix.empty()) return decl;
            if (suffix.empty() || !std::all_of(suffix.begin(), suffix.end(), ::isdigit)) continue;
            unsigned index = static_cast<unsigned>(std::stoul(suffix));
            if (index < decl.count && (!found || decl.prefix.size() > best.prefix.size())) {
                best = decl;
                found = true;
            }
        }
        if (!found) fail(line, "use of undeclared register '" + name + "'");
        return best;
    }

    std::optional<RegisterDecl> optional_declaration_for(const std::string &name) const {
        RegisterDecl best;
        bool found = false;
        for (const auto &decl : declarations_) {
            if (name.rfind(decl.prefix, 0) != 0) continue;
            std::string suffix = name.substr(decl.prefix.size());
            if (decl.count == 1 && suffix.empty()) return decl;
            if (suffix.empty() || !std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) { return std::isdigit(c); })) continue;
            unsigned index = static_cast<unsigned>(std::stoul(suffix));
            if (index < decl.count && (!found || decl.prefix.size() > best.prefix.size())) {
                best = decl;
                found = true;
            }
        }
        if (!found) return std::nullopt;
        return best;
    }

    void plan_register_layout() {
        std::unordered_map<std::string, std::size_t> by_name;
        std::unordered_map<std::string, unsigned> label_instruction;
        std::vector<std::pair<unsigned, std::string>> branches;
        unsigned instruction_index = 0;
        std::regex register_re(R"(%[A-Za-z_$][A-Za-z0-9_$.]*)");
        auto observe = [&](const std::string &text) {
            for (std::sregex_iterator it(text.begin(), text.end(), register_re), end; it != end; ++it) {
                std::string name = it->str();
                auto declaration = optional_declaration_for(name);
                if (!declaration) continue; // Special registers have no .reg declaration.
                auto known = by_name.find(name);
                if (known == by_name.end()) {
                    LiveInterval interval;
                    interval.name = name;
                    interval.type = declaration->type;
                    interval.begin = interval.end = instruction_index;
                    interval.references = 1;
                    interval.predicate = declaration->type == Type::None;
                    interval.width = is_pair_type(declaration->type) ? 2 : 1;
                    by_name[name] = intervals_.size();
                    intervals_.push_back(std::move(interval));
                } else {
                    intervals_[known->second].end = instruction_index;
                    ++intervals_[known->second].references;
                }
            }
        };
        for (const auto &item : program_.items) {
            if (item.label) label_instruction[*item.label] = instruction_index;
            if (!item.instruction) continue;
            const auto &inst = *item.instruction;
            if (inst.guard) observe(*inst.guard);
            for (const auto &operand : inst.operands) observe(operand);
            if (inst.mnemonic == "bra" && inst.operands.size() == 1) {
                branches.emplace_back(instruction_index, inst.operands[0]);
            }
            ++instruction_index;
        }

        // A lexical interval is insufficient for loop-carried execution: a value
        // whose sole textual use is in the middle of a loop is used again after
        // the backedge. Conservatively keep every interval touching a natural
        // loop live for the complete header-to-backedge region.
        for (const auto &[branch_index, target_name] : branches) {
            auto target = label_instruction.find(target_name);
            if (target == label_instruction.end() || target->second > branch_index) continue;
            unsigned loop_begin = target->second;
            unsigned loop_end = branch_index;
            for (auto &interval : intervals_) {
                if (interval.end < loop_begin || interval.begin > loop_end) continue;
                interval.begin = std::min(interval.begin, loop_begin);
                interval.end = std::max(interval.end, loop_end);
            }
        }

        std::vector<std::size_t> gpr_order, pred_order;
        for (std::size_t i = 0; i < intervals_.size(); ++i) {
            (intervals_[i].predicate ? pred_order : gpr_order).push_back(i);
        }
        auto by_start = [&](std::size_t a, std::size_t b) {
            if (intervals_[a].begin != intervals_[b].begin) return intervals_[a].begin < intervals_[b].begin;
            return intervals_[a].end < intervals_[b].end;
        };
        std::sort(gpr_order.begin(), gpr_order.end(), by_start);
        std::sort(pred_order.begin(), pred_order.end(), by_start);

        std::vector<std::size_t> active;
        auto spill_interval = [&](std::size_t interval_index) {
            auto &interval = intervals_[interval_index];
            // Assign LMEM offsets after register allocation, when every spilled
            // live range is known. This lets non-overlapping spills share a slot.
            registers_[interval.name] = {interval.type, 0, true, 0};
        };
        for (std::size_t index : gpr_order) {
            const auto &current = intervals_[index];
            active.erase(std::remove_if(active.begin(), active.end(), [&](std::size_t other) {
                return intervals_[other].end < current.begin;
            }), active.end());
            std::bitset<256> occupied;
            for (std::size_t other : active) {
                for (unsigned lane = 0; lane < intervals_[other].width; ++lane) {
                    occupied.set(intervals_[other].physical + lane);
                }
            }
            std::optional<uint16_t> selected;
            constexpr unsigned allocatable_gprs = 248; // R248-R255 are available to spill lowering.
            for (unsigned candidate = 0; candidate + current.width <= allocatable_gprs; ++candidate) {
                if (current.width == 2 && (candidate & 1u)) continue;
                bool free = true;
                for (unsigned lane = 0; lane < current.width; ++lane) free &= !occupied.test(candidate + lane);
                if (free) { selected = static_cast<uint16_t>(candidate); break; }
            }
            if (!selected) {
                auto victim = active.end();
                for (auto it = active.begin(); it != active.end(); ++it) {
                    if (intervals_[*it].width < current.width) continue;
                    if (victim == active.end() ||
                        intervals_[*it].references < intervals_[*victim].references ||
                        (intervals_[*it].references == intervals_[*victim].references &&
                         intervals_[*it].end > intervals_[*victim].end)) victim = it;
                }
                if (victim != active.end() &&
                    (intervals_[*victim].references < current.references ||
                     (intervals_[*victim].references == current.references && intervals_[*victim].end > current.end))) {
                    uint16_t reclaimed = intervals_[*victim].physical;
                    spill_interval(*victim);
                    active.erase(victim);
                    selected = reclaimed;
                } else {
                    spill_interval(index);
                    continue;
                }
            }
            intervals_[index].physical = *selected;
            physical_high_water_ = std::max<uint16_t>(
                physical_high_water_, static_cast<uint16_t>(*selected + current.width));
            registers_[current.name] = {current.type, *selected, false, 0};
            active.push_back(index);
        }

        std::vector<std::size_t> spilled;
        for (std::size_t index : gpr_order) {
            auto value = registers_.find(intervals_[index].name);
            if (value != registers_.end() && value->second.spilled) spilled.push_back(index);
        }
        std::sort(spilled.begin(), spilled.end(), by_start);
        active.clear();
        for (std::size_t index : spilled) {
            const auto &current = intervals_[index];
            active.erase(std::remove_if(active.begin(), active.end(), [&](std::size_t other) {
                return intervals_[other].end < current.begin;
            }), active.end());
            std::bitset<1024> occupied; // 4096 bytes of LMEM in four-byte words.
            for (std::size_t other : active) {
                const auto &value = registers_.at(intervals_[other].name);
                unsigned first_word = value.spill_offset / 4;
                for (unsigned lane = 0; lane < intervals_[other].width; ++lane) {
                    occupied.set(first_word + lane);
                }
            }
            std::optional<unsigned> selected_word;
            for (unsigned candidate = 0; candidate + current.width <= occupied.size(); ++candidate) {
                if (current.width == 2 && (candidate & 1u)) continue;
                bool free = true;
                for (unsigned lane = 0; lane < current.width; ++lane) {
                    free &= !occupied.test(candidate + lane);
                }
                if (free) {
                    selected_word = candidate;
                    break;
                }
            }
            if (!selected_word) {
                fail(0, "simultaneous local-memory spill requirement exceeds 4096 bytes per thread");
            }
            registers_.at(current.name).spill_offset = *selected_word * 4;
            active.push_back(index);
        }

        active.clear();
        for (std::size_t index : pred_order) {
            const auto &current = intervals_[index];
            active.erase(std::remove_if(active.begin(), active.end(), [&](std::size_t other) {
                return intervals_[other].end < current.begin;
            }), active.end());
            std::bitset<8> occupied;
            for (std::size_t other : active) occupied.set(intervals_[other].physical);
            std::optional<uint8_t> selected;
            for (unsigned candidate = 0; candidate < 8; ++candidate) {
                if (!occupied.test(candidate)) { selected = static_cast<uint8_t>(candidate); break; }
            }
            if (!selected) fail(0, "more than 8 simultaneously live predicates");
            intervals_[index].physical = *selected;
            predicate_high_water_ = std::max<uint8_t>(predicate_high_water_, *selected + 1);
            predicates_[current.name] = *selected;
            active.push_back(index);
        }

        if (pool_constants_) plan_constant_registers();
    }

    void plan_constant_registers() {
        std::vector<uint32_t> values;
        auto consider = [&](const std::string &operand, unsigned line) {
            std::string value = trim(operand);
            if (value.empty() || value.front() == '%' || value.front() == '[') return;
            uint32_t bits = static_cast<uint32_t>(parse_integer(value, line));
            if (std::find(values.begin(), values.end(), bits) == values.end()) values.push_back(bits);
        };
        for (const auto &item : program_.items) {
            if (!item.instruction) continue;
            const auto &inst = *item.instruction;
            std::string op = inst.mnemonic.substr(0, inst.mnemonic.find('.'));
            if (op == "bra" || op == "ret" || op == "mov" || op == "ld" || op == "st") continue;
            for (std::size_t i = 1; i < inst.operands.size(); ++i) {
                bool reducible_wide_mul = false;
                if (inst.mnemonic == "mul.wide.u32" && inst.operands.size() == 3) {
                    const std::string &other_operand = inst.operands[3 - i];
                    reducible_wide_mul = !other_operand.empty() && other_operand.front() == '%';
                }
                if (reducible_wide_mul &&
                    !inst.operands[i].empty() && inst.operands[i].front() != '%') {
                    uint32_t value = static_cast<uint32_t>(parse_integer(inst.operands[i], inst.line));
                    if (value != 0 && (value & (value - 1)) == 0) {
                        unsigned shift = 0;
                        while ((1u << shift) != value && shift < 31) ++shift;
                        consider(std::to_string(shift), inst.line);
                        continue;
                    }
                }
                consider(inst.operands[i], inst.line);
            }
        }
        std::bitset<256> ever_occupied;
        for (const auto &interval : intervals_) {
            if (interval.predicate) continue;
            auto value = registers_.find(interval.name);
            if (value != registers_.end() && value->second.spilled) continue;
            for (unsigned lane = 0; lane < interval.width; ++lane) ever_occupied.set(interval.physical + lane);
        }
        for (uint32_t value : values) {
            // Keep R248-R255 exclusively available for lowering temporaries and
            // spill reload/store sequences, matching the GPR allocator limit.
            for (unsigned candidate = 0; candidate < 248; ++candidate) {
                if (ever_occupied.test(candidate)) continue;
                ever_occupied.set(candidate);
                constant_registers_[value] = static_cast<uint16_t>(candidate);
                physical_high_water_ = std::max<uint16_t>(
                    physical_high_water_, static_cast<uint16_t>(candidate + 1));
                break;
            }
        }
    }

    RegisterValue reg(const std::string &name, unsigned line) {
        if (name.empty() || name.front() != '%') fail(line, "expected register, got '" + name + "'");
        auto it = registers_.find(name);
        if (it != registers_.end()) return it->second;
        RegisterDecl decl = declaration_for(name, line);
        if (decl.type == Type::None) fail(line, "predicate used as GPR: '" + name + "'");
        fail(line, "internal register-layout error for '" + name + "'");
    }

    uint16_t read_register(const std::string &name, unsigned line) {
        RegisterValue value = reg(name, line);
        if (!value.spilled) return value.physical;
        unsigned width = is_pair_type(value.type) ? 2 : 1;
        uint16_t loaded = temporary(line, width);
        for (unsigned lane = 0; lane < width; ++lane) {
            uint16_t address = temporary(line);
            emit(Opcode::LOADI, Type::None, address, 0, 0, value.spill_offset + lane * 4);
            emit(Opcode::LD, Type::B32, static_cast<uint16_t>(loaded + lane), address, 0, 0, 0, 3);
            release_temporary(address);
            ++stats_.spill_loads;
        }
        return loaded;
    }

    uint16_t destination_register(const std::string &name, unsigned line) {
        RegisterValue value = reg(name, line);
        if (!value.spilled) return value.physical;
        unsigned width = is_pair_type(value.type) ? 2 : 1;
        uint16_t result = temporary(line, width);
        pending_spill_writes_.emplace_back(name, result);
        return result;
    }

    void flush_spill_writes(unsigned line) {
        for (const auto &[name, physical] : pending_spill_writes_) {
            RegisterValue value = reg(name, line);
            unsigned width = is_pair_type(value.type) ? 2 : 1;
            for (unsigned lane = 0; lane < width; ++lane) {
                uint16_t address = temporary(line);
                emit(Opcode::LOADI, Type::None, address, 0, 0, value.spill_offset + lane * 4);
                emit(Opcode::ST, Type::B32, 0, address, physical + lane, 0, 0, 3,
                     active_guard_, active_guard_neg_);
                release_temporary(address);
                ++stats_.spill_stores;
            }
        }
        pending_spill_writes_.clear();
    }

    uint8_t predicate(const std::string &name, unsigned line) {
        auto it = predicates_.find(name);
        if (it != predicates_.end()) return it->second;
        RegisterDecl decl = declaration_for(name, line);
        if (decl.type != Type::None) fail(line, "GPR used as predicate: '" + name + "'");
        fail(line, "internal predicate-layout error for '" + name + "'");
    }

    uint16_t temporary(unsigned line, unsigned width = 1) {
        std::bitset<256> occupied;
        for (const auto &[unused, physical] : constant_registers_) {
            static_cast<void>(unused);
            occupied.set(physical);
        }
        for (const auto &interval : intervals_) {
            if (interval.predicate || current_instruction_ < interval.begin || current_instruction_ > interval.end) continue;
            auto allocated = registers_.find(interval.name);
            if (allocated != registers_.end() && allocated->second.spilled) continue;
            for (unsigned lane = 0; lane < interval.width; ++lane) occupied.set(interval.physical + lane);
        }
        for (uint16_t scratch : scratch_registers_) occupied.set(scratch);
        for (unsigned candidate = 0; candidate + width <= 256; ++candidate) {
            if (width == 2 && (candidate & 1u)) continue;
            bool free = true;
            for (unsigned lane = 0; lane < width; ++lane) free &= !occupied.test(candidate + lane);
            if (!free) continue;
            for (unsigned lane = 0; lane < width; ++lane) scratch_registers_.push_back(static_cast<uint16_t>(candidate + lane));
            physical_high_water_ = std::max<uint16_t>(
                physical_high_water_, static_cast<uint16_t>(candidate + width));
            return static_cast<uint16_t>(candidate);
        }
        fail(line, "no GPR available for lowering temporary; spilling is required");
    }

    void release_temporary(uint16_t physical, unsigned width = 1) {
        for (unsigned lane = 0; lane < width; ++lane) {
            uint16_t value = static_cast<uint16_t>(physical + lane);
            auto it = std::find(scratch_registers_.begin(), scratch_registers_.end(), value);
            if (it != scratch_registers_.end()) scratch_registers_.erase(it);
        }
    }

    static uint16_t ctrl(Type type, unsigned subop = 0, unsigned space = 0,
                         std::optional<uint8_t> guard = std::nullopt, bool neg = false) {
        uint16_t value = static_cast<uint16_t>(static_cast<unsigned>(type) << 3);
        value |= static_cast<uint16_t>((subop & 7u) << 8);
        value |= static_cast<uint16_t>((space & 7u) << 11);
        if (guard) {
            value |= static_cast<uint16_t>(*guard & 7u);
            value |= 1u << 15;
            if (neg) value |= 1u << 14;
        }
        return value;
    }

    std::pair<std::optional<uint8_t>, bool> guard_of(const PtxInstruction &ptx) {
        if (!ptx.guard) return {std::nullopt, false};
        return {predicate(*ptx.guard, ptx.line), ptx.guard_neg};
    }

    void emit(uint16_t opcode, Type type, uint16_t dest = 0, uint16_t src1 = 0,
              uint32_t src2 = 0, uint32_t imm = 0, unsigned subop = 0,
              unsigned space = 0, std::optional<uint8_t> guard = std::nullopt,
              bool guard_neg = false) {
        code_.push_back({opcode, ctrl(type, subop, space, guard, guard_neg), dest, src1, src2, imm, std::nullopt});
    }

    static uint64_t parse_integer(const std::string &text, unsigned line) {
        if (text.size() == 10 && (text.rfind("0f", 0) == 0 || text.rfind("0F", 0) == 0)) {
            try { return std::stoull(text.substr(2), nullptr, 16); } catch (const std::exception &) {}
        }
        try {
            std::size_t consumed = 0;
            long long signed_value = std::stoll(text, &consumed, 0);
            if (consumed == text.size()) return static_cast<uint64_t>(signed_value);
        } catch (const std::exception &) {}
        try {
            std::size_t consumed = 0;
            unsigned long long value = std::stoull(text, &consumed, 0);
            if (consumed == text.size()) return static_cast<uint64_t>(value);
        } catch (const std::exception &) {}
        throw CompileError("line " + std::to_string(line) + ": unsupported immediate '" + text + "'");
    }

    uint16_t source(const std::string &operand, unsigned line) {
        if (!operand.empty() && operand.front() == '%') return read_register(operand, line);
        uint32_t value = static_cast<uint32_t>(parse_integer(operand, line));
        auto constant = constant_registers_.find(value);
        if (constant != constant_registers_.end()) return constant->second;
        uint16_t temp = temporary(line);
        emit(Opcode::LOADI, Type::None, temp, 0, 0, value);
        return temp;
    }

    static std::vector<std::string> mnemonic_parts(const std::string &mnemonic) {
        std::vector<std::string> parts;
        std::stringstream ss(mnemonic);
        std::string item;
        while (std::getline(ss, item, '.')) parts.push_back(item);
        return parts;
    }

    static Type final_type(const std::vector<std::string> &parts) {
        if (parts.empty()) throw CompileError("empty mnemonic");
        return parse_type(parts.back());
    }

    static void require_operands(const PtxInstruction &ptx, std::size_t count) {
        if (ptx.operands.size() != count) {
            throw CompileError("line " + std::to_string(ptx.line) + ": '" + ptx.mnemonic +
                               "' expects " + std::to_string(count) + " operands");
        }
    }

    void lower(const PtxInstruction &ptx) {
        const auto parts = mnemonic_parts(ptx.mnemonic);
        const std::string &op = parts.front();
        auto [guard, guard_neg] = guard_of(ptx);
        active_guard_ = guard;
        active_guard_neg_ = guard_neg;

        if (op == "ld" && parts.size() >= 3 && parts[1] == "param") {
            require_operands(ptx, 2);
            uint16_t dst = destination_register(ptx.operands[0], ptx.line);
            std::string name = unbracket(ptx.operands[1]);
            auto parameter = params_.find(name);
            if (parameter == params_.end()) fail(ptx.line, "unknown kernel parameter '" + name + "'");
            Type type = final_type(parts);
            uint16_t addr = temporary(ptx.line);
            emit(Opcode::LOADI, Type::None, addr, 0, 0, parameter->second.offset);
            emit(Opcode::LD, type == Type::B64 ? Type::U32 : type, dst, addr, 0, 0, 0, 4, guard, guard_neg);
            if (type == Type::B64) {
                emit(Opcode::LOADI, Type::None, addr, 0, 0, parameter->second.offset + 4);
                emit(Opcode::LD, Type::U32, static_cast<uint16_t>(dst + 1), addr, 0, 0, 0, 4,
                     guard, guard_neg);
            }
            return;
        }

        if (op == "mov") {
            require_operands(ptx, 2);
            uint16_t dst = destination_register(ptx.operands[0], ptx.line);
            Type type = final_type(parts);
            const std::string &src = ptx.operands[1];
            static const std::unordered_map<std::string, uint16_t> specials = {
                {"%tid.x",0x0100},{"%ntid.x",0x0101},{"%ctaid.x",0x0102},{"%nctaid.x",0x0103},{"%laneid",0x0104},
                {"%tid.y",0x0110},{"%ntid.y",0x0111},{"%ctaid.y",0x0112},{"%nctaid.y",0x0113},
                {"%tid.z",0x0120},{"%ntid.z",0x0121},{"%ctaid.z",0x0122},{"%nctaid.z",0x0123}
            };
            auto special = specials.find(src);
            if (special != specials.end()) {
                emit(Opcode::CPY, Type::U32, dst, special->second, 0, 0, 0, 0, guard, guard_neg);
            } else if (!src.empty() && src.front() == '%') {
                uint16_t source_reg = read_register(src, ptx.line);
                emit(Opcode::CPY, type, dst, source_reg, 0, 0, 0, 0, guard, guard_neg);
            } else {
                uint64_t value = parse_integer(src, ptx.line);
                if (type == Type::B64) emit(Opcode::LOADI64, Type::None, dst, 0, static_cast<uint32_t>(value >> 32), static_cast<uint32_t>(value), 0, 0, guard, guard_neg);
                else emit(Opcode::LOADI, Type::None, dst, 0, 0, static_cast<uint32_t>(value), 0, 0, guard, guard_neg);
            }
            return;
        }

        if (op == "bra") {
            require_operands(ptx, 1);
            MachineInstruction inst;
            inst.opcode = guard ? Opcode::BRX : Opcode::BR;
            inst.ctrl = ctrl(Type::None, 0, 0, guard, guard_neg);
            inst.branch_label = ptx.operands[0];
            code_.push_back(std::move(inst));
            return;
        }
        if (op == "ret") {
            require_operands(ptx, 0);
            emit(Opcode::HALT, Type::None, 0, 0, 0, 0, 0, 0, guard, guard_neg);
            return;
        }

        if (op == "setp") {
            require_operands(ptx, 3);
            if (parts.size() < 3) fail(ptx.line, "invalid setp mnemonic");
            static const std::unordered_map<std::string, unsigned> comparisons = {
                {"eq",0},{"ne",1},{"lt",2},{"le",3},{"gt",4},{"ge",5}
            };
            auto cmp = comparisons.find(parts[1]);
            if (cmp == comparisons.end()) fail(ptx.line, "unsupported comparison '" + parts[1] + "'");
            uint8_t dst = predicate(ptx.operands[0], ptx.line);
            uint16_t a = source(ptx.operands[1], ptx.line);
            uint16_t b = source(ptx.operands[2], ptx.line);
            emit(Opcode::CMPP, final_type(parts), dst, a, b, 0, cmp->second, 0, guard, guard_neg);
            return;
        }

        if ((op == "ld" || op == "st") && parts.size() >= 3 && parts[1] == "global") {
            require_operands(ptx, 2);
            Type type = final_type(parts);
            if (op == "ld") {
                uint16_t dst = destination_register(ptx.operands[0], ptx.line);
                uint16_t addr = read_register(unbracket(ptx.operands[1]), ptx.line);
                emit(Opcode::LD, type, dst, addr, 0, 0, 0, 0, guard, guard_neg);
            } else {
                uint16_t addr = read_register(unbracket(ptx.operands[0]), ptx.line);
                uint16_t value = read_register(ptx.operands[1], ptx.line);
                emit(Opcode::ST, type, 0, addr, value, 0, 0, 0, guard, guard_neg);
            }
            return;
        }

        if (op == "mul" && parts.size() >= 3 && parts[1] == "wide") {
            require_operands(ptx, 3);
            uint16_t dst = destination_register(ptx.operands[0], ptx.line);
            std::string multiplicand = ptx.operands[1];
            std::string multiplier = ptx.operands[2];
            if (!multiplicand.empty() && multiplicand.front() != '%' &&
                !multiplier.empty() && multiplier.front() == '%') {
                std::swap(multiplicand, multiplier);
            }
            uint16_t a = source(multiplicand, ptx.line);
            bool reduced = false;
            if (!multiplier.empty() && multiplier.front() != '%') {
                uint32_t value = static_cast<uint32_t>(parse_integer(multiplier, ptx.line));
                if (value == 0) {
                    emit(Opcode::LOADI, Type::None, dst, 0, 0, 0, 0, 0, guard, guard_neg);
                    reduced = true;
                } else if ((value & (value - 1)) == 0) {
                    unsigned shift = 0;
                    while ((1u << shift) != value && shift < 31) ++shift;
                    uint16_t amount = source(std::to_string(shift), ptx.line);
                    emit(Opcode::SHL, Type::U32, dst, a, amount, 0, 0, 0, guard, guard_neg);
                    reduced = true;
                }
            }
            if (!reduced) {
                uint16_t b = source(multiplier, ptx.line);
                emit(Opcode::MUL, Type::U32, dst, a, b, 0, 0, 0, guard, guard_neg);
            }
            emit(Opcode::LOADI, Type::None, static_cast<uint16_t>(dst + 1), 0, 0, 0, 0, 0,
                 guard, guard_neg);
            return;
        }

        if (op == "add" && final_type(parts) == Type::B64) {
            require_operands(ptx, 3);
            uint16_t dst = destination_register(ptx.operands[0], ptx.line);
            uint16_t a = read_register(ptx.operands[1], ptx.line);
            uint16_t b = read_register(ptx.operands[2], ptx.line);
            emit(Opcode::ADD, Type::U32, dst, a, b, 0, 0, 0, guard, guard_neg);
            emit(Opcode::LOADI, Type::None, static_cast<uint16_t>(dst + 1), 0, 0, 0, 0, 0,
                 guard, guard_neg);
            return;
        }

        static const std::unordered_map<std::string, uint16_t> binary_ops = {
            {"add",Opcode::ADD},{"sub",Opcode::SUB},{"mul",Opcode::MUL},
            {"and",Opcode::AND},{"or",Opcode::OR},{"xor",Opcode::XOR},
            {"shl",Opcode::SHL},{"shr",Opcode::SHR}
        };
        auto binary = binary_ops.find(op);
        if (binary != binary_ops.end()) {
            require_operands(ptx, 3);
            uint16_t dst = destination_register(ptx.operands[0], ptx.line);
            uint16_t a = source(ptx.operands[1], ptx.line);
            uint16_t b = source(ptx.operands[2], ptx.line);
            Type encoded_type = final_type(parts);
            // PTX spells left shift as shl.b32, while the released AEC model's
            // canonical legal type is u32/s32. SHL has identical bit semantics.
            if (op == "shl" && encoded_type == Type::B32) encoded_type = Type::U32;
            emit(binary->second, encoded_type, dst, a, b, 0, 0, 0, guard, guard_neg);
            return;
        }

        if (op == "mad" || op == "fma") {
            require_operands(ptx, 4);
            uint16_t dst = destination_register(ptx.operands[0], ptx.line);
            uint16_t a = source(ptx.operands[1], ptx.line);
            uint16_t b = source(ptx.operands[2], ptx.line);
            uint16_t c = source(ptx.operands[3], ptx.line);
            emit(op == "mad" ? Opcode::MAD : Opcode::FMA, final_type(parts), dst, a, b, c, 0, 0, guard, guard_neg);
            return;
        }

        fail(ptx.line, "unsupported instruction '" + ptx.mnemonic + "'");
    }
};

std::string read_file(const std::string &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw CompileError("cannot open input file '" + path + "'");
    std::ostringstream out;
    out << stream.rdbuf();
    return out.str();
}

void write_u32_le(std::ofstream &out, uint32_t value) {
    char bytes[4] = {static_cast<char>(value), static_cast<char>(value >> 8),
                     static_cast<char>(value >> 16), static_cast<char>(value >> 24)};
    out.write(bytes, 4);
}

void write_binary(const std::string &path, const std::vector<MachineInstruction> &code) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw CompileError("cannot create output file '" + path + "'");
    for (const auto &inst : code) {
        uint32_t w0 = inst.imm;
        uint32_t w1 = inst.src2;
        uint32_t w2 = (static_cast<uint32_t>(inst.dest) << 16) | inst.src1;
        uint32_t w3 = (static_cast<uint32_t>(inst.opcode) << 16) | inst.ctrl;
        write_u32_le(out, w0); write_u32_le(out, w1); write_u32_le(out, w2); write_u32_le(out, w3);
    }
    if (!out) throw CompileError("failed while writing output file '" + path + "'");
}

std::string json_escape(const std::string &value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        if (c == '"') out << "\\\"";
        else if (c == '\\') out << "\\\\";
        else if (c == '\n') out << "\\n";
        else if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
        else out << c;
    }
    return out.str();
}

void write_report(const std::string &path, const std::string &input, const std::string &output,
                  const std::string &opt, const CompileStats &stats, std::size_t aec_count) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) throw CompileError("cannot create report file '" + path + "'");
    out << "{\n"
        << "  \"status\": \"ok\",\n"
        << "  \"input\": \"" << json_escape(input) << "\",\n"
        << "  \"output\": \"" << json_escape(output) << "\",\n"
        << "  \"opt_level\": \"" << json_escape(opt) << "\",\n"
        << "  \"num_ptx_instructions\": " << stats.ptx_instructions << ",\n"
        << "  \"num_aec_instructions\": " << aec_count << ",\n"
        << "  \"num_basic_blocks\": " << stats.basic_blocks << ",\n"
        << "  \"num_virtual_registers\": " << stats.virtual_registers << ",\n"
        << "  \"num_physical_registers\": " << stats.physical_registers << ",\n"
        << "  \"num_predicates\": " << stats.predicates << ",\n"
        << "  \"spills\": {\"loads\": " << stats.spill_loads
        << ", \"stores\": " << stats.spill_stores << "},\n"
        << "  \"passes\": {\"dce\": " << (opt != "O0" ? "true" : "false")
        << ", \"constant_folding\": " << (opt != "O0" ? "true" : "false")
        << ", \"cse\": " << (opt == "O2" ? "true" : "false")
        << ", \"mad_fusion\": " << (opt == "O2" ? "true" : "false")
        << ", \"licm\": " << (opt == "O2" ? "true" : "false")
        << ", \"load_hoisting\": " << (opt == "O2" ? "true" : "false")
        << ", \"branch_simplification\": " << (opt != "O0" ? "true" : "false")
        << ", \"scheduler\": \"" << (opt == "O2" ? "list" : "source") << "\"},\n"
        << "  \"warnings\": []\n"
        << "}\n";
}

struct Options {
    std::string input;
    std::string output;
    std::optional<std::string> report;
    std::string opt_level = "O0";
};

Options parse_options(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-o") {
            if (++i >= argc) throw CompileError("-o requires a path");
            options.output = argv[i];
        } else if (arg == "--report") {
            if (++i >= argc) throw CompileError("--report requires a path");
            options.report = argv[i];
        } else if (arg == "-O0" || arg == "-O1" || arg == "-O2") {
            options.opt_level = arg.substr(1);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "usage: aec-cc input.ptx [-O0|-O1|-O2] -o output.aecbin [--report report.json]\n";
            std::exit(0);
        } else if (!arg.empty() && arg.front() == '-') {
            throw CompileError("unknown option '" + arg + "'");
        } else if (options.input.empty()) {
            options.input = arg;
        } else {
            throw CompileError("unexpected positional argument '" + arg + "'");
        }
    }
    if (options.input.empty()) throw CompileError("missing input PTX file");
    if (options.output.empty()) throw CompileError("missing required -o output path");
    return options;
}

} // namespace

int main(int argc, char **argv) {
    try {
        Options options = parse_options(argc, argv);
        Program program = parse_ptx(read_file(options.input));
        unsigned input_instruction_count = 0;
        for (const auto &item : program.items) if (item.instruction) ++input_instruction_count;
        if (options.opt_level == "O2") {
            program = eliminate_ptx_common_expressions(std::move(program));
            program = hoist_loop_invariants(std::move(program));
            program = reduce_loop_address_strength(std::move(program));
            program = rotate_guarded_loops(std::move(program));
            program = hoist_loop_invariants(std::move(program));
            program = unroll_rotated_loop_by_four(std::move(program));
            program = fuse_ptx_multiply_add(std::move(program));
        }
        auto [code, stats] = Lowerer(program, options.opt_level == "O2").run();
        if (options.opt_level == "O2") code = eliminate_common_expressions(std::move(code));
        if (options.opt_level != "O0") {
            code = fold_constants_and_branches(std::move(code));
            code = eliminate_unreachable_code(std::move(code));
            code = eliminate_redundant_branches(std::move(code));
        }
        if (options.opt_level != "O0") {
            code = eliminate_dead_code(std::move(code));
            code = eliminate_redundant_branches(std::move(code));
            code = eliminate_dead_code(std::move(code));
        }
        if (options.opt_level == "O2") {
            code = fuse_multiply_add(std::move(code));
            code = eliminate_dead_code(std::move(code));
            code = schedule_instructions(std::move(code));
        }
        stats.ptx_instructions = input_instruction_count;
        stats.basic_blocks = count_basic_blocks(code);
        stats.spill_loads = stats.spill_stores = 0;
        for (const auto &inst : code) {
            unsigned space = (inst.ctrl >> 11) & 7u;
            if (space == 3 && inst.opcode == Opcode::LD) ++stats.spill_loads;
            if (space == 3 && inst.opcode == Opcode::ST) ++stats.spill_stores;
        }
        if (code.empty()) throw CompileError("kernel generated no AEC instructions");
        write_binary(options.output, code);
        if (options.report) write_report(*options.report, options.input, options.output, options.opt_level, stats, code.size());
        return 0;
    } catch (const CompileError &error) {
        std::cerr << "aec-cc: error: " << error.what() << '\n';
        return 1;
    } catch (const std::exception &error) {
        std::cerr << "aec-cc: internal error: " << error.what() << '\n';
        return 2;
    }
}
