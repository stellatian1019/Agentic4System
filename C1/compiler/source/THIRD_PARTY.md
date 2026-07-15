# Third-party and AI assistance disclosure

The compiler and disassembler sources have no project-specific third-party
source dependencies; they use the C++17 standard library. The submitted
`compiler/aec-cc` Linux x86-64 entry was built with the GNU toolchain and
statically contains the corresponding GNU C Library, libstdc++, and libgcc
runtime portions. libstdc++ and libgcc are distributed under GPLv3 with the GCC
Runtime Library Exception; the GNU C Library is distributed under LGPLv2.1 or
later. The non-evaluation disassembler entry is a POSIX shell launcher.

OpenAI Codex was used during development to assist with repository analysis,
implementation, test construction, debugging, and documentation. The resulting
source is included in full and is intended to be reviewed, explained, and
maintained by the submitting team, as required by the competition rules.

The released `aec-precise-linux-x86_64` Golden Model was used only as an external
development-time functional validator. It is not included in this submission.
