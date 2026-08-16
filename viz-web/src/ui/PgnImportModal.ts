/**
 * Kepler-64 PGN / FEN Import Modal (2.2)
 * Paste PGN text, drop a .pgn file, or load an arbitrary FEN position.
 */

import { Chess } from 'chess.js';

export interface PgnImportCallbacks {
  onImportPgn: (pgn: string) => void;
  onImportFen: (fen: string) => void;
}

export class PgnImportModal {
  private container: HTMLElement;
  private callbacks: PgnImportCallbacks;

  constructor(callbacks: PgnImportCallbacks) {
    this.callbacks = callbacks;
    this.container = document.createElement('div');
    this.container.className = 'modal-backdrop';
    this.container.style.display = 'none';
    document.body.appendChild(this.container);
    this.render();
  }

  public open(): void {
    this.resetStatus();
    this.container.style.display = 'flex';
  }

  public close(): void {
    this.container.style.display = 'none';
  }

  private resetStatus(): void {
    const status = this.container.querySelector('#import-status');
    if (status) {
      status.textContent = '';
      status.className = 'import-status';
    }
  }

  private render(): void {
    this.container.innerHTML = `
      <div class="modal-content">
        <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:var(--space-md); border-bottom:var(--rule-thin) solid var(--color-rule); padding-bottom:var(--space-sm);">
          <div>
            <span class="badge-tag">CUSTOM MATCH ANALYZER</span>
            <h3 style="font-size:1.4rem;">Import a Game or Position</h3>
          </div>
          <button id="import-close-btn" class="action-secondary" style="padding:4px 8px;">✕ Close</button>
        </div>

        <p style="font-size:0.88rem; color:var(--color-muted); margin-bottom:var(--space-md);">
          Paste PGN from Chess.com / Lichess, drop a <code>.pgn</code> file, or load a single FEN.
          Kepler-64 will sweep every ply for gravitational potential, tidal stress, and King Roche index.
        </p>

        <label class="import-label" for="import-pgn-text">PGN TEXT</label>
        <textarea id="import-pgn-text" class="import-textarea" rows="8" spellcheck="false"
          placeholder="[Event \"...\"]&#10;[White \"...\"]&#10;[Black \"...\"]&#10;&#10;1. e4 e5 2. Nf3 ..."></textarea>
        <div class="import-file-row">
          <input id="import-pgn-file" type="file" accept=".pgn,.txt" class="import-file-input" />
          <span class="import-hint">or drop a .pgn file onto the text area</span>
        </div>

        <div class="import-divider"></div>

        <label class="import-label" for="import-fen-text">FEN POSITION</label>
        <input id="import-fen-text" class="import-fen-input" spellcheck="false"
          placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" />

        <div id="import-status" class="import-status"></div>

        <div class="modal-actions">
          <button id="import-fen-btn" class="action-secondary">Analyze Position</button>
          <button id="import-pgn-btn" class="action-primary">Analyze PGN</button>
        </div>
      </div>
    `;

    this.container.querySelector('#import-close-btn')?.addEventListener('click', () => this.close());
    this.container.addEventListener('click', (e) => {
      if (e.target === this.container) this.close();
    });

    this.container.querySelector('#import-pgn-btn')?.addEventListener('click', () => this.submitPgn());
    this.container.querySelector('#import-fen-btn')?.addEventListener('click', () => this.submitFen());

    const textarea = this.container.querySelector('#import-pgn-text') as HTMLTextAreaElement | null;
    const fileInput = this.container.querySelector('#import-pgn-file') as HTMLInputElement | null;

    fileInput?.addEventListener('change', () => {
      const file = fileInput.files?.[0];
      if (!file || !textarea) return;
      const reader = new FileReader();
      reader.onload = () => {
        textarea.value = String(reader.result ?? '');
      };
      reader.readAsText(file);
    });

    textarea?.addEventListener('dragover', (e) => {
      e.preventDefault();
    });

    textarea?.addEventListener('drop', (e) => {
      e.preventDefault();
      const file = e.dataTransfer?.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        textarea.value = String(reader.result ?? '');
      };
      reader.readAsText(file);
    });
  }

  private setStatus(message: string, isError: boolean): void {
    const status = this.container.querySelector('#import-status');
    if (!status) return;
    status.textContent = message;
    status.className = `import-status ${isError ? 'import-status-error' : 'import-status-ok'}`;
  }

  private submitPgn(): void {
    const textarea = this.container.querySelector('#import-pgn-text') as HTMLTextAreaElement | null;
    const raw = (textarea?.value ?? '').trim();
    if (!raw) {
      this.setStatus('Paste or drop a PGN first.', true);
      return;
    }

    try {
      const chess = new Chess();
      chess.loadPgn(raw);
      if (chess.history().length === 0) {
        this.setStatus('No moves found in the PGN.', true);
        return;
      }
    } catch (err) {
      this.setStatus(`Invalid PGN: ${err instanceof Error ? err.message : String(err)}`, true);
      return;
    }

    this.setStatus('Imported. Sweeping gravitational trajectory…', false);
    this.callbacks.onImportPgn(raw);
    this.close();
  }

  private submitFen(): void {
    const input = this.container.querySelector('#import-fen-text') as HTMLInputElement | null;
    const fen = (input?.value ?? '').trim();
    if (!fen) {
      this.setStatus('Enter a FEN string first.', true);
      return;
    }

    try {
      const chess = new Chess(fen);
      const normalized = chess.fen(); // normalize + validate in one pass
      this.setStatus('Imported. Rendering gravitational topography…', false);
      this.callbacks.onImportFen(normalized);
      this.close();
    } catch (err) {
      this.setStatus(`Invalid FEN: ${err instanceof Error ? err.message : String(err)}`, true);
    }
  }
}
