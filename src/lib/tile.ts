export type RecognizedTile = {
  classId: number;
  label: string;
  confidence: number;
  bbox: [number, number, number, number];
  centerX: number;
  centerY: number;
  rotationDegrees?: number | null;
};

export function parseTileInput(value: string): string[] {
  return value
    .replace(/[\[\]"']/g, ' ')
    .split(/[\s,，]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

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

export function separateHandAndFuro(tiles: RecognizedTile[]): {
  handTiles: RecognizedTile[];
  furoGroups: RecognizedTile[][];
} {
  if (tiles.length <= 2) {
    return { handTiles: [...tiles], furoGroups: [] };
  }

  const medianHeight = median(tiles.map(getTileHeight));
  const rowTolerance = Math.max(8, medianHeight * 0.5);

  const rows = clusterIntoRows(tiles, rowTolerance);

  if (rows.length <= 1) {
    const sorted = [...tiles].sort((a, b) => a.centerX - b.centerX);
    return { handTiles: sorted, furoGroups: [] };
  }

  rows.sort((a, b) => b.length - a.length);

  const handRow = rows[0];
  const furoGroups = rows.slice(1).map((row) =>
    [...row].sort((a, b) => a.centerX - b.centerX),
  );

  const handTiles = [...handRow].sort((a, b) => a.centerX - b.centerX);

  return { handTiles, furoGroups };
}

function clusterIntoRows(tiles: RecognizedTile[], tolerance: number): RecognizedTile[][] {
  const sorted = [...tiles].sort((a, b) => a.centerY - b.centerY);
  const rows: RecognizedTile[][] = [];
  let currentRow: RecognizedTile[] = [sorted[0]];
  let rowCenter = sorted[0].centerY;

  for (let i = 1; i < sorted.length; i++) {
    const tile = sorted[i];
    if (Math.abs(tile.centerY - rowCenter) <= tolerance) {
      currentRow.push(tile);
      rowCenter = currentRow.reduce((s, t) => s + t.centerY, 0) / currentRow.length;
    } else {
      rows.push(currentRow);
      currentRow = [tile];
      rowCenter = tile.centerY;
    }
  }

  rows.push(currentRow);
  return rows;
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
