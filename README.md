# napari-zarr-io

[![License](https://img.shields.io/pypi/l/napari-zarr-io.svg?color=green)](https://github.com/napari/napari-zarr-io/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-zarr-io.svg?color=green)](https://pypi.org/project/napari-zarr-io)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-zarr-io.svg?color=green)](https://python.org)
[![tests](https://github.com/manzt/napari-zarr-io/workflows/tests/badge.svg)](https://github.com/manzt/napari-zarr-io/actions)

A more feature-complete zarr reader plugin for napari.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/docs/plugins/index.html
-->

## Installation

You can install `napari-zarr-io` via [pip]:

    pip install napari-zarr-io
    
## Description
This napari reader plugin is meant to handle opening both zarr `group` and `array` from a local `zarr.DirectoryStore`.
It will **not** return a reader for the zarr [multiscales extension](https://github.com/zarr-developers/zarr-specs/issues/50) (check out [`ome-zarr-py`](https://github.com/ome/ome-zarr-py) for this use case) or remote zarr arrays.

### Example zarr store

```bash
examples/channels_astronaut.zarr
├── .zgroup
├── blue
│   ├── .zarray
│   ├── 0.0
│   └── 1.0
├── green
│   ├── .zarray
│   ├── 0.0
│   └── 1.0
└── red
    ├── .zarray
    ├── 0.0
    └── 1.0
```

### Open arrays within `zarr.Group` as separate image layers  

```bash
$ napari examples/channels_astronaut.zarr # adds red, green, blue image layers
```

### Open a `zarr.Array` as an image layer

```bash
$ napari examples/channels_astronaut.zarr/blue # adds one image layer
```


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-zarr-io" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/manzt/napari-zarr-io/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/