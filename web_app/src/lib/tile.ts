export type RecognizedTile = {
  classId: number;
  label: string;
  confidence: number;
  bbox: [number, number, number, number];
  centerX: number;
  centerY: number;
};

export function isMahjongTileCode(label: string): boolean {
  return /^(?:[1-9]|0)[mpsz]$/.test(label);
}

export function sortTilesLeftToRight(tiles: RecognizedTile[]): RecognizedTile[] {
  return [...tiles].sort((left, right) => {
    if (left.centerX !== right.centerX) {
      return left.centerX - right.centerX;
    }

    return left.centerY - right.centerY;
  });
}

export function toOrderedTileLabels(tiles: RecognizedTile[]): string[] {
  return sortTilesLeftToRight(tiles).map((tile) => tile.label);
}

export function countRedDora(tileLabels: string[]): number {
  return tileLabels.filter((label) => label === '0m' || label === '0p' || label === '0s').length;
}

export function hasUnsupportedTiles(tileLabels: string[]): boolean {
  return tileLabels.some((label) => !isMahjongTileCode(label));
}
