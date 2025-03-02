/// <reference types="cypress" />

import './commands'

// コンポーネントテスト用の設定をここに追加
declare global {
  namespace Cypress {
    interface Chainable {
      mount: typeof mount
    }
  }
}

// React コンポーネントのマウント関数
function mount(component: React.ReactElement) {
  const root = document.createElement('div')
  root.setAttribute('data-cy-root', '')
  document.body.appendChild(root)

  return cy.mount(component, { root })
}
