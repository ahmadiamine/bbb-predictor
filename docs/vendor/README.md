# Vendored RDKit

`RDKit_minimal.js` and `RDKit_minimal.wasm` are the official
[@rdkit/rdkit](https://www.npmjs.com/package/@rdkit/rdkit) WebAssembly build
(version 2026.9.2), vendored so the site has no external dependency.

RDKit is released under the BSD 3-Clause license. See
https://github.com/rdkit/rdkit/blob/master/license.txt

They are not modified. To update: `npm pack @rdkit/rdkit` and copy
`package/dist/RDKit_minimal.{js,wasm}` into this folder.
