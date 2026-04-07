# CC-IDF

## Command

```bash
python infer_mesh.py --config-json ckpt/doorhandle/config.json --rot2-deg -30.0 --rot3-deg 25.0 --iso-level -0.01
```

## Environment

Tested with:
- Python 3.10
- numpy 2.2.6
- torch 2.6.0+cu124
- scikit-image 0.25.2

## Input

The program requires:
- a JSON config file
- three object checkpoints
- one q-network checkpoint
- rotation angles for object 2 and object 3
- an iso-surface level

## Output

The extracted mesh is saved automatically as:

```bash
out.obj
```

in the current working directory.

## Notes

- The iso-level must satisfy: abs(iso-level) <= 0.04