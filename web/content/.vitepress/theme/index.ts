import DefaultTheme from 'vitepress/theme'
import type { Theme } from 'vitepress'
import { h, nextTick, onMounted, onUnmounted, watch } from 'vue'
import { useRoute } from 'vitepress'
import mediumZoom from 'medium-zoom'
import TextbookHome from './TextbookHome.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  Layout: () => h(DefaultTheme.Layout),
  enhanceApp({ app }) { app.component('TextbookHome', TextbookHome) },
  setup() {
    const route = useRoute()
    let zoom: ReturnType<typeof mediumZoom> | undefined
    const attach = async () => {
      await nextTick()
      zoom?.detach()
      zoom = mediumZoom('.vp-doc img:not(a img)', { background: 'var(--vp-c-bg)', margin: 24 })
    }
    onMounted(attach)
    watch(() => route.path, attach)
    onUnmounted(() => zoom?.detach())
  }
} satisfies Theme
