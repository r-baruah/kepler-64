import './style.css';
import { ObservatoryApp } from './ui/ObservatoryApp';

const appEl = document.querySelector<HTMLDivElement>('#app');
if (appEl) {
  new ObservatoryApp(appEl);
}
