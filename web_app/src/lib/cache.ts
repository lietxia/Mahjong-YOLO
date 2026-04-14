export const ASSET_CACHE_PREFIX = 'mahjong-yolo-phase5';

export type AssetCacheStatus = {
  supported: boolean;
  enabled: boolean;
  message: string;
};

function toErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : '未知错误';
}

export async function registerAssetCacheServiceWorker(): Promise<AssetCacheStatus> {
  if (typeof window === 'undefined' || typeof navigator === 'undefined') {
    return {
      supported: false,
      enabled: false,
      message: '当前环境不是浏览器，未注册静态资源缓存策略。',
    };
  }

  if (!('serviceWorker' in navigator)) {
    return {
      supported: false,
      enabled: false,
      message: '当前浏览器不支持 Service Worker，将仅依赖浏览器默认 HTTP 缓存。',
    };
  }

  if (!import.meta.env.PROD) {
    return {
      supported: true,
      enabled: false,
      message: '开发模式下不注册 Service Worker；生产构建会启用静态资源缓存钩子。',
    };
  }

  try {
    await navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js`, {
      scope: import.meta.env.BASE_URL,
    });

    return {
      supported: true,
      enabled: true,
      message: '已启用最小静态资源缓存：应用壳、模型清单和已访问过的模型资源会进入 Service Worker 缓存。',
    };
  } catch (error) {
    return {
      supported: true,
      enabled: false,
      message: `Service Worker 注册失败：${toErrorMessage(error)}`,
    };
  }
}

export async function clearRegisteredAssetCaches(): Promise<string> {
  const messages: string[] = [];

  if ('caches' in globalThis) {
    const keys = await caches.keys();
    const matchedKeys = keys.filter((key) => key.startsWith(ASSET_CACHE_PREFIX));
    await Promise.all(matchedKeys.map((key) => caches.delete(key)));
    messages.push(matchedKeys.length > 0 ? `已清理 ${matchedKeys.length} 个应用缓存。` : '未检测到应用缓存。');
  } else {
    messages.push('当前浏览器不支持 Cache Storage。');
  }

  if (typeof navigator !== 'undefined' && 'serviceWorker' in navigator) {
    const registrations = await navigator.serviceWorker.getRegistrations();
    const matches = registrations.filter((registration) => {
      const scriptUrl = registration.active?.scriptURL ?? registration.waiting?.scriptURL ?? registration.installing?.scriptURL ?? '';
      return scriptUrl.endsWith('/sw.js');
    });

    await Promise.all(matches.map((registration) => registration.unregister()));
    if (matches.length > 0) {
      messages.push('已注销当前页面使用的 Service Worker，请刷新页面以重新建立缓存。');
    }
  }

  return messages.join(' ');
}
