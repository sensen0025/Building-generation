import argparse,json
from pathlib import Path
from .data import build_dataset,class_contract,atomic_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--limit',type=int);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 old=Path('/home/linux/redstone_ai/dataset');new=Path('/data/redstone_v4')
 contract=class_contract(old/'BLOCK_VOCAB_FULL.txt',old/'v2/block_map.npy')
 sources=[str(p) for folder in [old/'litematics',new/'sources'] for p in folder.iterdir() if p.suffix.lower() in ('.litematic','.schem','.schematic','.nbt','.mcstructure')]
 historic=[str(old/'litematics'/s) for s in (old/'valid_files_list.txt').read_text().splitlines()]
 historic += [str(new/'sources'/r['file']) for r in json.loads((new/'work/ingest_manifest_ok.json').read_text())]
 if a.smoke:
  sources=sorted(str(p) for p in (new/'sources').glob('*.litematic'))[:a.limit or 12];historic=[]
 m=build_dataset(a.out,sources,contract,historic,max_tiles=256,limit=None if a.smoke else a.limit)
 m['smoke_only']=a.smoke;atomic_json(Path(a.out)/'manifest.json',m)
 print(json.dumps({k:v for k,v in m.items() if k!='samples'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
