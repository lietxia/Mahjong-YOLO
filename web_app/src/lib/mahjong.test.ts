import { describe, expect, it } from 'vitest';
import { calculateMahjongScore } from './mahjong';

describe('mahjong score adapter', () => {
  it('returns incomplete when tile count is not 14', () => {
    const result = calculateMahjongScore(['1m', '2m'], {
      fieldWind: 'east',
      seatWind: 'east',
      agariType: 'tsumo',
      riichi: false,
      ippatsu: false,
      agariIndex: 1,
      doraIndicators: [],
      uraIndicators: [],
    });

    expect(result.status).toBe('incomplete');
  });

  it('returns ready for a 14-tile hand input', () => {
    const result = calculateMahjongScore(
      ['1m', '2m', '3m', '4m', '5m', '6m', '2p', '3p', '4p', '7s', '8s', '9s', '5z', '5z'],
      {
        fieldWind: 'east',
        seatWind: 'east',
        agariType: 'ron',
        riichi: false,
        ippatsu: false,
        agariIndex: 13,
        doraIndicators: [],
        uraIndicators: [],
      },
    );

    expect(result.status).toBe('ready');
    expect(result.normalizedTiles).toHaveLength(14);
    expect(result.result).toBeDefined();
  });
});
