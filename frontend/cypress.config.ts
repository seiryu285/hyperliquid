// 型定義がなくてもエラーを出さないようにする
declare global {
  namespace Cypress {
    interface PluginEvents {
      [key: string]: any;
    }
    interface PluginConfigOptions {
      [key: string]: any;
    }
  }
}

// 型定義を使用せずに設定を記述
const config = {
  e2e: {
    setupNodeEvents(on: any, config: any) {
      return config;
    },
    baseUrl: 'http://localhost:3000',
    supportFile: 'cypress/support/e2e.ts',
    specPattern: 'cypress/e2e/**/*.cy.{js,jsx,ts,tsx}',
    video: false,
    screenshotOnRunFailure: true,
    defaultCommandTimeout: 10000,
    viewportWidth: 1280,
    viewportHeight: 720,
  },
};

export default config;
