declare module 'date-fns' {
  export function format(date: Date | number, format: string, options?: any): string;
  export function addDays(date: Date | number, amount: number): Date;
  export function subDays(date: Date | number, amount: number): Date;
  export function startOfDay(date: Date | number): Date;
  export function endOfDay(date: Date | number): Date;
  export function parseISO(dateString: string): Date;
  export function formatDistance(date: Date | number, baseDate: Date | number, options?: any): string;
  export function isValid(date: any): boolean;
}
