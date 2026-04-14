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
  const selectedTiles = selectPrimaryHorizontalRow(tiles);

  return [...selectedTiles].sort((left, right) => {
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

export function selectPrimaryHorizontalRow(tiles: RecognizedTile[]): RecognizedTile[] {
  if (tiles.length <= 2) {
    return [...tiles];
  }

  const medianHeight = median(tiles.map(getTileHeight));
  const rowTolerance = Math.max(8, medianHeight * 0.5);

  let bestGroup: RecognizedTile[] = [];
  let bestConfidenceSum = Number.NEGATIVE_INFINITY;
  let bestSpread = Number.POSITIVE_INFINITY;

  for (const anchor of tiles) {
    const group = tiles.filter((tile) => Math.abs(tile.centerY - anchor.centerY) <= rowTolerance);
    const confidenceSum = group.reduce((sum, tile) => sum + tile.confidence, 0);
    const spread = getVerticalSpread(group);

    const isBetterGroup =
      group.length > bestGroup.length ||
      (group.length === bestGroup.length && confidenceSum > bestConfidenceSum) ||
      (group.length === bestGroup.length && confidenceSum === bestConfidenceSum && spread < bestSpread);

    if (isBetterGroup) {
      bestGroup = group;
      bestConfidenceSum = confidenceSum;
      bestSpread = spread;
    }
  }

  return bestGroup.length > 0 ? [...bestGroup] : [...tiles];
}

function getTileHeight(tile: RecognizedTile): number {
  return Math.max(0, tile.bbox[3] - tile.bbox[1]);
}

function getVerticalSpread(tiles: RecognizedTile[]): number {
  if (tiles.length === 0) {
    return Number.POSITIVE_INFINITY;
  }

  const centers = tiles.map((tile) => tile.centerY);
  return Math.max(...centers) - Math.min(...centers);
}

function median(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }

  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);

  if (sorted.length % 2 === 0) {
    return (sorted[middle - 1] + sorted[middle]) / 2;
  }

  return sorted[middle];
}
